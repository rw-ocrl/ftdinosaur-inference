"""Slot attention.

This implementation is adapted from the oclf library under Apache License 2.0.

See: https://github.com/amazon-science/object-centric-learning-framework/
Original file: https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/perceptual_grouping.py
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn


class RandomSlotInitialization(nn.Module):
    """Random slot initialization with potentially learnt mean and stddev."""

    def __init__(
        self,
        slot_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
        logsigma_init: Callable[[torch.Tensor], None] = nn.init.xavier_uniform_,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = slot_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, slot_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, slot_dim))

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int, num_slots: Optional[int] = None) -> torch.Tensor:
        num_slots = num_slots if num_slots is not None else self.n_slots
        mu = self.slots_mu.expand(batch_size, num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, num_slots, -1)
        return mu + sigma * torch.randn_like(mu)


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError(
                "Key, value, query dimensions must be divisible by number of heads."
            )
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)

        if isinstance(ff_mlp, nn.Module):
            self.ff_mlp = ff_mlp
        else:
            self.ff_mlp = nn.Identity()

    def step(self, slots, k, v):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        q = q * self.scale

        dots = torch.einsum("bihd,bjhd->bihj", q, k)
        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn).flatten(-2, -1)

        slots = self.gru(
            updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim)
        )
        slots = slots.reshape(updates.shape[0], -1, self.dim)

        slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def forward(self, inputs: torch.Tensor, slots: torch.Tensor):
        b, n, _ = inputs.shape

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v)

        return slots, attn


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        feature_transform: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            slot_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `slot_dim` is used.
            n_heads: Number of heads slot attention uses.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            feature_transform: Optional module applied to the features before slot attention.
            use_projection_bias: Whether to use biases in key, value, query projections.
        """
        super().__init__()
        self.slot_attention = SlotAttention(
            dim=slot_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
        )
        self.feature_transform = (
            feature_transform if feature_transform else nn.Identity()
        )

    def forward(
        self,
        features: torch.Tensor,
        initial_slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply slot attention based perceptual grouping.

        Args:
            features: Features used for grouping.

        Returns:
            The grouped features.
        """
        features = self.feature_transform(features)
        slots, attn_masks = self.slot_attention(features, initial_slots)

        return slots, attn_masks
