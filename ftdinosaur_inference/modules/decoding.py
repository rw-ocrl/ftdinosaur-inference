"""Decoders.

This implementation is adapted from the oclf library under Apache License 2.0.

See: https://github.com/amazon-science/object-centric-learning-framework/
Original file: https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/decoding.py
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn


class PatchDecoder(nn.Module):
    """Decoder that takes slot representations and reconstructs patches.

    Args:
        slot_dim: Dimension of slot representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of slots, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from slot to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
        top_k: Number of slots to decode per-position. Selects the top-k slots according to `mask`.
    """

    def __init__(
        self,
        slot_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_input_dim: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.top_k = top_k
        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(slot_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = slot_dim

        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, decoder_input_dim) * 0.02
        )

    def select_top_k(
        self, slots: torch.Tensor, masks: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k objects per position according to their values in masks."""
        # slots: [batch_dims] x n_slots x n_positions x dims
        # masks: [batch_dims] x n_slots x n_positions

        batch_dims = slots.shape[:-3]
        slots = slots.flatten(0, -4)  # Flatten batch dimensions
        batch_size, _, n_positions, dims = slots.shape

        with torch.no_grad():
            masks = masks.detach().flatten(0, -3)  # Flatten batch dimensions
            masks = masks.transpose(1, 2).flatten(0, 1)  # b s p -> (b p) s
            idxs = torch.topk(masks, k=k, dim=1, sorted=False).indices
            idxs = idxs.unflatten(0, (batch_size, n_positions)).transpose(
                1, 2
            )  # (b p) k -> b k p
            idxs = idxs.unsqueeze(-1).expand((-1, -1, -1, dims))  # b k p -> b k p d

        slots = torch.gather(
            slots, dim=1, index=idxs
        )  # Select top_k slots per position
        slots = slots.unflatten(0, batch_dims)
        idxs = idxs.unflatten(0, batch_dims)

        return slots, idxs

    def restore_masks_after_top_k(
        self, masks: torch.Tensor, idxs: torch.Tensor, n_masks: int
    ) -> torch.Tensor:
        """Fill masks with zeros for all non-top-k objects."""
        # masks: [batch_dims] x top_k_objects x n_positions
        # idxs: [batch_dims] x top_k_objects x n_positions x dims
        batch_dims = masks.shape[:-2]
        masks_all = torch.zeros(
            *batch_dims, n_masks, masks.shape[-1], device=masks.device
        )
        masks_all.scatter_(dim=1, index=idxs[..., 0], src=masks)
        return masks_all

    def forward(
        self,
        slots: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ):
        assert slots.ndim == 3, "Slots must have shape (batch, num_slots, slot_dim)"

        batch_size, num_slots = slots.shape[:-1]
        num_orig_slots = num_slots
        slots = slots.flatten(0, 1)

        if self.inp_transform is not None:
            slots = self.inp_transform(slots)

        # Broadcast slots over patches.
        slots = slots.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT.
        slots = slots + self.pos_embed

        top_k = top_k if top_k is not None else self.top_k
        should_do_top_k = top_k is not None and top_k > 0
        if should_do_top_k:
            if masks is None:
                raise ValueError("Need to pass `masks` for top_k.")
            assert (
                masks.ndim == 3
            ), "Masks must have shape (batch, num_slots, num_patches)"
            slots, top_k_idxs = self.select_top_k(
                slots.unflatten(0, (batch_size, num_slots)), masks, top_k
            )
            num_slots = top_k
            slots = slots.flatten(0, 1)

        output = self.decoder(slots)
        output = output.unflatten(0, (batch_size, num_slots))

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)

        masks = alpha.softmax(dim=1)  # Softmax over slots
        reconstruction = torch.sum(decoded_patches * masks, dim=1)
        masks = masks.squeeze(-1)

        if should_do_top_k:
            masks = self.restore_masks_after_top_k(masks, top_k_idxs, num_orig_slots)

        return reconstruction, masks
