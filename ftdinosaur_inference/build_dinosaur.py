"""Build pre-trained DINOSAUR models.

Design heavily inspired by the timm library.
"""

from typing import Any, Callable, Dict, Final, List, Optional

import torch

from ftdinosaur_inference import utils
from ftdinosaur_inference.modules import dinosaur


def _cfg(url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "url": url,
        "input_size": (3, 224, 224),
        "input_mean": utils.IMAGENET_MEAN,
        "input_std": utils.IMAGENET_STD,
        **kwargs,
    }


# Model configs.
CONFIGS: Final[Dict[str, Any]] = {
    "dinosaur_small_patch14_224_topk3.coco_dv2_ft_s7_300k": _cfg(
        url="https://huggingface.co/mseitzer/ftdinosaur/resolve/main/dinosaur_small_patch14_224_topk3.coco_dv2_ft_s7_300k-fbc7a7c7.pth"
    ),
    "dinosaur_small_patch14_518_topk3.coco_dv2_ft_s7_300k+10k": _cfg(
        url="https://huggingface.co/mseitzer/ftdinosaur/resolve/main/dinosaur_small_patch14_518_topk3.coco_dv2_ft_s7_300k+10k-d23af290.pth",
        input_size=(3, 518, 518),
    ),
    "dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k": _cfg(
        url="https://huggingface.co/mseitzer/ftdinosaur/resolve/main/dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k-b617dc95.pth",
    ),
    "dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k": _cfg(
        url="https://huggingface.co/mseitzer/ftdinosaur/resolve/main/dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k-15486674.pth",
        input_size=(3, 518, 518),
    ),
}

# Registry for builder functions.
_ENTRY_POINTS: Dict[str, Callable] = {}


def register_model():
    """Decorator for registering builder functions."""

    def wrapper(func):
        assert func.__name__.startswith("build_")
        model_name = func.__name__.replace("build_", "", 1)
        assert model_name not in _ENTRY_POINTS
        _ENTRY_POINTS[model_name] = func
        return func

    return wrapper


def get_config(model_name: str) -> Optional[Dict[str, Any]]:
    model_base, _, weights = model_name.partition(".")
    if weights == "":
        # Use first weights that fit model base
        avail = list(k for k in CONFIGS if k.partition(".")[0] == model_base)
        if len(avail) == 0:
            return None
        weights = avail[0].partition(".")[-1]
    model_key = f"{model_base}.{weights}" if len(weights) > 0 else model_base
    return CONFIGS.get(model_key)


def _build_dinosaur(
    variant: str, pretrained: bool = False, **kwargs
) -> dinosaur.DINOSAUR:
    if pretrained:
        config = get_config(variant)
        if not config or not config.get("url"):
            raise ValueError(f"Model {variant} has no pre-trained checkpoint defined.")
        url = config["url"]
    else:
        url = None

    model = dinosaur.build(**kwargs)

    if pretrained and url:
        state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True)
        model.load_state_dict(state_dict)

    return model


@register_model()
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


@register_model()
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


@register_model()
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


@register_model()
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
    return list(key for key, cfg in CONFIGS.items() if cfg.get("url") is not None)


def build(model_name: str, pretrained: bool = True, **kwargs) -> dinosaur.DINOSAUR:
    """Build DINOSAUR model variant and potentially load pre-trained checkpoint."""
    model_base, _, weights = model_name.partition(".")
    if model_base not in _ENTRY_POINTS:
        raise ValueError(f"Unknown model {model_name}")

    return _ENTRY_POINTS[model_base](model_name, pretrained, **kwargs)


def build_preprocessing(model_name: str, **kwargs) -> torch.nn.Module:
    """Build pre-processing pipeline for model."""
    config = get_config(model_name)
    if not config:
        raise ValueError(f"Model {model_name} has no config defined.")

    return utils.build_preprocessing(
        config["input_size"][1:], config["input_mean"], config["input_std"], **kwargs
    )
