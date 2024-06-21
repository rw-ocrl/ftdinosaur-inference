import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as tvt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_preprocessing(
    input_size: Union[int, Tuple[int, int]],
    input_mean: Tuple[float, float, float],
    input_std: Tuple[float, float, float],
    center_crop: bool = False,
) -> torch.nn.Module:
    """Build preprocessing pipeline for input images.

    Make sure `input_size` matches the resolution the model was trained on.
    """
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.Resize(
                size=(input_size[0], input_size[1]),
                interpolation=tvt.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            tvt.CenterCrop(input_size) if center_crop else torch.nn.Identity(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=input_mean, std=input_std),
        ]
    )


def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to, either size or (height, width).
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, H, W) where H, W are height and width of the image.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


def soft_masks_to_one_hot(masks: torch.Tensor) -> torch.Tensor:
    """Convert probability masks to one-hot masks, using argmax.

    Args:
        masks: Masks of shape (B, C, H, W), where C is the number of masks.
    """
    masks_argmax = masks.argmax(dim=1)[:, None]
    idxs = torch.arange(
        masks.shape[1], device=masks_argmax.device, dtype=masks_argmax.dtype
    )
    return masks_argmax == idxs[None, :, None, None]


def convert_checkpoint_from_oclf(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert checkpoint from OCLF framework to state dict loadable with this codebase."""
    prefix_map = {
        "models.feature_extractor.model": "encoder",
        "models.conditioning": "slot_init",
        "models.perceptual_grouping.positional_embedding.layers.1": "slot_attention.feature_transform",
        "models.perceptual_grouping.slot_attention": "slot_attention.slot_attention",
        "models.object_decoder": "decoder",
    }

    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        for prefix, new_prefix in prefix_map.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, new_prefix, 1)
                state_dict[new_key] = value
                break

    return state_dict
