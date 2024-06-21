"""DINOSAUR model."""

from functools import partial
from typing import Any, Dict, Optional

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
        slot_init: Optional[torch.Tensor] = None,
        decode: bool = True,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run model.

        Args:
            images: Tensor with images of shape (batch, channels, height, width).
            num_slots: Number of slots to use. If not given, use model default.
            initial_slots: Optional tensor with slot init of (batch, num_slots, slot_dim).
                If not given, sample random slots.
            decode: If true, run decoder and return decoder outputs.
            decoder_kwargs: Optional keyword arguments to the decoder.

        """
        if images.ndim != 4:
            raise ValueError("Images must have shape (batch, channels, height, width)")

        if slot_init is None:
            slot_init = self.slot_init(len(images), num_slots)
        else:
            if slot_init.ndim != 3:
                raise ValueError(
                    "`slot_init` must have shape (batch, num_slots, slot_dim)"
                )
            if len(slot_init) != len(images):
                raise ValueError(
                    f"`slot_init` must match image batch size of {len(images)}, "
                    f"but got {len(slot_init)}"
                )

        features = self.encoder(images)
        slots, slot_masks = self.slot_attention(features, slot_init)
        output = {
            "features": features,
            "slot_init": slot_init,
            "slots": slots,
            "slot_masks": slot_masks,
        }

        if decode:
            pred_features, decoder_masks = self.decoder(
                slots,
                masks=slot_masks,
                **decoder_kwargs if decoder_kwargs else {},
            )
            output["pred_features"] = pred_features
            output["masks"] = decoder_masks

        return output


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
