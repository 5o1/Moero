import torch
import math
from typing import Tuple, Sequence

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[-2] <= data.size(-2) and 0 < shape[-1] <= data.size(-1)):
        raise ValueError("Invalid shapes.")

    x_start = math.floor(data.size(-2) / 2) + math.ceil(-shape[-2] / 2)
    x_end = math.floor(data.size(-2) / 2) + math.ceil(shape[-2] / 2)

    y_start = math.floor(data.size(-1) / 2) + math.ceil(-shape[-1] / 2)
    y_end = math.floor(data.size(-1) / 2) + math.ceil(shape[-1] / 2)
    return data[..., x_start: x_end, y_start: y_end]

def make_center_mask(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    if not (0 < shape[-2] <= data.size(-2) and 0 < shape[-1] <= data.size(-1)):
        raise ValueError("Invalid shapes.")

    x_start = math.floor(data.size(-2) / 2) + math.ceil(-shape[-2] / 2)
    x_end = math.floor(data.size(-2) / 2) + math.ceil(shape[-2] / 2)

    y_start = math.floor(data.size(-1) / 2) + math.ceil(-shape[-1] / 2)
    y_end = math.floor(data.size(-1) / 2) + math.ceil(shape[-1] / 2)

    mask = torch.zeros_like(data)
    mask[..., x_start: x_end, y_start: y_end] = 1
    return mask

def center_crop_to_smallest(
    *tensors: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    minimum_height = min(t.size(-2) for t in tensors)
    minimum_width = min(t.size(-1) for t in tensors)

    results = []
    for tensor in tensors:
        if tensor.size(-2) != minimum_height or tensor.size(-1) != minimum_width:
            results.append(center_crop(tensor, (minimum_height, minimum_width)))
        else:
            results.append(tensor)
    return tuple(results)