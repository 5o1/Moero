import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift

def itok(input: torch.Tensor, dim = [-2, -1]) -> torch.Tensor: 
    input = ifftshift(input, dim = dim)
    input = fftn(input, dim = dim)
    input = fftshift(input, dim = dim)
    return input

def ktoi(input: torch.Tensor, dim = [-2, -1]) -> torch.Tensor:
    input = ifftshift(input, dim = dim)
    input = ifftn(input, dim = dim)
    input = fftshift(input, dim = dim)
    return input

def complex2chan(input: torch.Tensor, dim = 1) -> torch.Tensor:
    input = torch.cat(
        [input.real, input.imag],
        dim = dim
    )
    return input

def chan2complex(input: torch.Tensor, dim = 1) -> torch.Tensor:
    input = input.chunk(2, dim)
    input = torch.complex(input[0], input[1])
    return input

def mask_extractor(input: torch.Tensor, eps = 1e-13, return_bool = False) -> torch.Tensor:
    """
    input: [batch, channel, height, width]
    output: [batch, 1, height, width]
    """

    mask = input.sum(dim = 1, keepdim = True)
    mask = mask.abs() > eps
    if return_bool:
        return mask
    mask = mask.float()
    return mask