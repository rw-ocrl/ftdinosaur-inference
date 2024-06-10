"""Build pre-trained DINOSAUR models."""

import torch

from ftdinosaur_inference.modules import dinosaur

CHECKPOINT_URLS = {}


def _build_dinosaur(
    variant: str, pretrained: bool = False, **kwargs
) -> dinosaur.DINOSAUR:
    model = dinosaur.build(**kwargs)
    if pretrained:
        if variant not in CHECKPOINT_URLS:
            raise ValueError(f"Variant {variant} has no pretrained checkpoint defined.")
        url = CHECKPOINT_URLS[variant]
        state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True)
        model.load_state_dict(state_dict)

    return model


def build_dinosaur_small_patch14_224_topk3(
    pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-S/14 backbone, trained on 224x224 images with topk-decoding."""
    model_args = dict(
        encoder_type="small14",
        slot_dim=256,
        num_slots=7,
        num_patches=256,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        "dinosaur_small_patch14_224_topk3.coco_s7_300k",
        pretrained,
        **dict(model_args, **kwargs),
    )


def build_dinosaur_small_patch14_518_topk3(
    pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-S/14 backbone, trained on 518x518 images with topk-decoding."""
    model_args = dict(
        encoder_type="small14",
        slot_dim=256,
        num_slots=7,
        num_patches=1369,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        "dinosaur_small_patch14_224_topk3.coco_s7_300k",
        pretrained,
        **dict(model_args, **kwargs),
    )
