"""Build pre-trained DINOSAUR models."""

from typing import Callable, Dict, Final, List

import torch

from ftdinosaur_inference import utils
from ftdinosaur_inference.modules import dinosaur

CHECKPOINT_URLS: Final[Dict[str, str]] = {
    "dinosaur_small_patch14_224_topk3.coco_s7_300k": None,
    "dinosaur_small_patch14_518_topk3.coco_s7_300k_10k": None,
    "dinosaur_base_patch14_224_topk3.coco_s7_300k": None,
    "dinosaur_base_patch14_518_topk3.coco_s7_300k_10k": None,
}

_INPUT_SIZE_BY_MODEL: Dict[str, int] = {}
_ENTRY_POINTS: Dict[str, Callable] = {}


def register_model(input_size: int):
    def wrapper(func):
        assert func.__name__.startswith("build_")
        model_name = func.__name__.replace("build_", "", 1)
        assert model_name not in _ENTRY_POINTS
        _ENTRY_POINTS[model_name] = func
        _INPUT_SIZE_BY_MODEL[model_name] = input_size
        return func

    return wrapper


def _build_dinosaur(
    variant: str, pretrained: bool = False, **kwargs
) -> dinosaur.DINOSAUR:
    model = dinosaur.build(**kwargs)
    if pretrained:
        model_base, _, weights = variant.partition(".")
        if weights == "":
            # Use first weights that fit model key
            avail = list(
                k for k in CHECKPOINT_URLS if k.partition(".")[0] == model_base
            )
            weights = avail[0].partition(".")[0]
        model_key = f"{model_base}.{weights}"
        if not CHECKPOINT_URLS.get(model_key):
            raise ValueError(f"Model {variant} has no pretrained checkpoint defined.")
        url = CHECKPOINT_URLS[model_key]
        state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True)
        model.load_state_dict(state_dict)

    return model


@register_model(input_size=224)
def build_dinosaur_small_patch14_224_topk3(
    variant: str, pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-S/14 backbone, trained on 224x224 images, top-k decoding."""
    model_args = dict(
        encoder_type="small14",
        slot_dim=256,
        num_slots=7,
        num_patches=256,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        variant,
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model(input_size=518)
def build_dinosaur_small_patch14_518_topk3(
    variant: str, pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-S/14 backbone, trained on 518x518 images (hi-res), top-k decoding."""
    model_args = dict(
        encoder_type="small14",
        slot_dim=256,
        num_slots=7,
        num_patches=1369,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        variant,
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model(input_size=224)
def build_dinosaur_base_patch14_224_topk3(
    variant: str, pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-B/14 backbone, trained on 224x224 images, top-k decoding."""
    model_args = dict(
        encoder_type="base14",
        slot_dim=256,
        num_slots=7,
        num_patches=256,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        variant,
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model(input_size=518)
def build_dinosaur_base_patch14_518_topk3(
    variant: str, pretrained: bool = True, **kwargs
) -> dinosaur.DINOSAUR:
    """Build DINOSAUR with ViT-B/14 backbone, trained on 518x518 images (hi-res), top-k decoding."""
    model_args = dict(
        encoder_type="base14",
        slot_dim=256,
        num_slots=7,
        num_patches=1369,
        decoder_top_k=3,
    )
    return _build_dinosaur(
        variant,
        pretrained,
        **dict(model_args, **kwargs),
    )


def list_models() -> List[str]:
    """Return list of models."""
    return list(_ENTRY_POINTS)


def list_checkpoints() -> List[str]:
    """Return list of models variants with pre-trained checkpoints."""
    return list(key for key, url in CHECKPOINT_URLS.items() if url is not None)


def build(model_name: str, pretrained: bool = True, **kwargs) -> dinosaur.DINOSAUR:
    """Build DINOSAUR model variant and potentially load pre-trained checkpoint."""
    model_base, _, weights = model_name.partition(".")
    if model_base not in _ENTRY_POINTS:
        raise ValueError(f"Unknown model {model_name}")

    return _ENTRY_POINTS[model_base](model_name, pretrained, **kwargs)


def build_preprocessing(model_name: str, **kwargs) -> torch.nn.Module:
    """Build preprocessing pipeline for model."""
    input_sizes = list(
        size for key, size in _INPUT_SIZE_BY_MODEL.items() if model_name.startswith(key)
    )
    if len(input_sizes) == 0:
        raise ValueError(f"No input size defined for model {model_name}")

    return utils.build_preprocessing(input_sizes[0], **kwargs)
