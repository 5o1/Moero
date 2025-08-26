"""
Transform the output file of model predict into Cmr25 format.
"""


import torch
from math import floor, ceil
from typing import Tuple
from einops import rearrange

from .import mrimodality as mm

def get_slice_index(nslice: int):
    if nslice < 3:
        islice = list(range(nslice)) # All of the 2 slices
    else:
        center = matlab_round(nslice / 2) - 1
        islice = list(range(center - 1, center + 1))  # Central 2 slices

    return islice

def get_frame_index(nframe: int, modality: mm.MriModality):
    if isinstance(modality, mm.BlackBlood) or nframe == 1: # Use only the first frame
        iframe = 0
    elif isinstance(modality, (mm.Mapping, mm.T1rho)):
        iframe = list(range(nframe)) # Use all frames
    else:
        iframe = list(range(min(3, nframe))) # Use the first 3 frames
    return iframe

def rsos(tensor: torch.Tensor, dim: int = -3, pnorm=2):
    return (tensor**pnorm).abs().sum(dim, keepdim=False)**(1/pnorm)

def matlab_round(x):
    if x > 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

def center_crop(tensor: torch.Tensor, target_shape: Tuple[ int | None]):
    """
    Parameters:
        tensor: torch.Tensor: The input tensor to crop.
        target_shape: Tuple[int | None]: A tuple specifying the desired shape for each dimension of the tensor.
            A value of None for a dimension means no cropping for that dimension.

    Returns:
        Returns the cropped tensor.
    """
    assert tensor.ndim == len(target_shape)

    indice = []
    for i in range(tensor.ndim):
        if target_shape[i] is None:
            indice.append(slice(None))
        else:
            center = floor(tensor.size(i)/2)
            start =  center + ceil(-target_shape[i]/2)
            end = center + ceil(target_shape[i]/2)
            indice.append(slice(start, end))
    return tensor[tuple(indice)]


def transform4ranking(tensor: torch.Tensor, modality: mm.MriModality) -> torch.Tensor:
    # [slice height weight] for mapping t1w t2w
    # [frame slice height weight] for others
    if tensor.ndim == 3:
        nslice, h, w = tensor.shape
        nframe = 1
    elif tensor.ndim == 4:
        nframe, nslice, h, w = tensor.shape
    else:
        raise ValueError("Dim of input must be 4d or 5d.")

    islice = get_slice_index(nslice)
    iframe = get_frame_index(nframe, modality)

    # tensor = rsos(tensor)
    tensor = tensor.abs()
    tensor = tensor.float()

    if tensor.ndim == 3:
        tensor = tensor[islice, :, :]
        tensor = center_crop(tensor, (None, matlab_round(h / 3), matlab_round(w / 2)))
        tensor = rearrange(tensor, "s h w -> h w s")
    else:
        iframe, islice = torch.meshgrid(torch.as_tensor(iframe), torch.as_tensor(islice), indexing="ij")
        tensor = tensor[iframe, islice, :, :]
        tensor = center_crop(tensor, (None, None, matlab_round(h / 3), matlab_round(w / 2)))
        tensor = rearrange(tensor, "t s h w -> h w s t")
    return tensor