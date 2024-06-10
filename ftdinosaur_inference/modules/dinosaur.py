"""DINOSAUR model."""

from functools import partial
from typing import Dict, Optional

import torch
from torch import nn

from ftdinosaur_inference.modules import vit
from ftdinosaur_inference.modules.decoding import PatchDecoder
from ftdinosaur_inference.modules.helpers import build_mlp, build_two_layer_mlp
from ftdinosaur_inference.modules.slot_attention import (
    RandomSlotInitialization,
    SlotAttentionGrouping,
)


class DINOSAUR(nn.Module):
    """DINOSAUR model."""

    def __init__(
        self,
        encoder: nn.Module,
        slot_init: nn.Module,
        slot_attention: nn.Module,
        decoder: nn.Module,
    ):
        """
        Args:
            encoder: Module that creates features from the input image.
            slot_init: Module that creates initial slots
            slot_attention: Module that creates slots from features.
            decoder: Module that predicts features from slots.
        """
        super().__init__()
        self.encoder = encoder
        self.slot_init = slot_init
        self.slot_attention = slot_attention
        self.decoder = decoder

    def forward(
        self,
        images: torch.Tensor,
        num_slots: Optional[int] = None,
        decoder_top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert (
            images.ndim == 4
        ), "Input must have shape (batch, channels, height, width)"

        features = self.encoder(images)
        initial_slots = self.slot_init(len(features), num_slots)
        slots, slot_masks = self.slot_attention(features, initial_slots)
        pred_features, decoder_masks = self.decoder(
            slots, masks=slot_masks, top_k=decoder_top_k
        )

        return {
            "features": features,
            "slots": slots,
            "slot_masks": slot_masks,
            "pred_features": pred_features,
            "masks": decoder_masks,
        }


def build(
    encoder_type: str,
    slot_dim: int,
    num_slots: int,
    num_patches: int,
    decoder_top_k: Optional[int] = None,
) -> DINOSAUR:
    if encoder_type == "small14":
        encoder_fn = vit.build_vit_small_patch14_dinov2
    elif encoder_type == "base14":
        encoder_fn = vit.build_vit_base_patch14_dinov2
    else:
        raise ValueError(f"Unsupported encoder type {encoder_type}")

    encoder = encoder_fn(
        dynamic_img_size=True,
        no_final_norm=True,
        skip_prefix_tokens=True,  # Remove CLS token from output
    )
    feature_dim = encoder.embed_dim

    return DINOSAUR(
        encoder=encoder,
        slot_init=RandomSlotInitialization(slot_dim, num_slots),
        slot_attention=SlotAttentionGrouping(
            slot_dim,
            slot_dim,
            ff_mlp=build_two_layer_mlp(
                slot_dim,
                slot_dim,
                hidden_dim=4 * slot_dim,
                initial_layer_norm=True,
                residual=True,
            ),
            feature_transform=build_two_layer_mlp(
                feature_dim,
                slot_dim,
                hidden_dim=2 * feature_dim,
                initial_layer_norm=True,
            ),
        ),
        decoder=PatchDecoder(
            slot_dim,
            feature_dim,
            num_patches,
            decoder=partial(build_mlp, features=[2048, 2048, 2048]),
            top_k=decoder_top_k,
        ),
    )
