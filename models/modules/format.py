import torch
from torch.nn import functional as F
from typing import Tuple, List
import math
from utils.complex import interpolate

class Format4Unet2d(torch.nn.Module):
    def __init__(self, ndownsample: int = 3, is_pad = True, is_norm = True, is_resize = True):
        super().__init__()
        self.ndownsample = ndownsample
        self.is_pad = is_pad
        self.is_norm = is_norm
        self.is_resize = is_resize

        self.bias = None
        self.scale = None
        self.pad_cache = None
        self.raw_shape = None

    def norm(self, tensor: torch.Tensor, is_fit: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_complex(tensor):
            tensor = torch.view_as_real(tensor)
            dims = tuple(range(1, tensor.ndim))
            if is_fit:
                self.bias = tensor.mean(dim=dims, keepdim=True)
                self.scale = tensor.std(dim=dims, keepdim=True)
            tensor = (tensor - self.bias) / self.scale
            tensor = torch.view_as_complex(tensor)
        else:
            dims = tuple(range(1, tensor.ndim))
            if is_fit:
                self.bias = tensor.mean(dim=dims, keepdim=True)
                self.scale = tensor.std(dim=dims, keepdim=True)
            tensor = (tensor - self.bias) / self.scale
        return tensor, self.scale, self.bias

    def norm_adjoint(self, tensor: torch.Tensor, scale: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        if scale is None:
            scale = self.scale
        if bias is None:
            bias = self.bias
        if torch.is_complex(tensor):
            tensor = torch.view_as_real(tensor)
            tensor = tensor * scale + bias
            tensor = torch.view_as_complex(tensor)
            return tensor
        else:
            return tensor * scale + bias

    def pad(self, tensor: torch.Tensor, is_fit: bool = True) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        h, w = tensor.size(-2), tensor.size(-1)
        div = 2 ** self.ndownsample

        w_mult = ((w - 1) | (div - 1)) + 1
        h_mult = ((h - 1) | (div - 1)) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        tensor = F.pad(tensor, w_pad + h_pad)

        if is_fit:
            self.pad_cache = (h_pad, w_pad, h_mult, w_mult)

        return tensor, (h_pad, w_pad, h_mult, w_mult)

    def pad_adjoint(
            self, tensor: torch.Tensor,
            h_pad: List[int] = None,
            w_pad: List[int] = None,
            h_mult: int = None,
            w_mult: int = None
        ) -> torch.Tensor:

        if h_pad is None:
            h_pad = self.pad_cache[0]
        if w_pad is None:
            w_pad = self.pad_cache[1]
        if h_mult is None:
            h_mult = self.pad_cache[2]
        if w_mult is None:
            w_mult = self.pad_cache[3]

        return tensor[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]


    def resize(self, tensor: torch.Tensor, is_fit: bool = True):
        if is_fit:
            self.raw_shape = tensor.shape
        npts = tensor.size(-2) * tensor.size(-1)
        hw = math.ceil(math.sqrt(npts))
        if tensor.size(-2) != hw or tensor.size(-1) != hw:
            tensor = interpolate(tensor.view(-1, *tensor.shape[-3:]), size=(hw, hw), mode='bilinear', align_corners=False).view(*tensor.shape[:-3], -1, hw, hw)

        return tensor, self.raw_shape
    
    def resize_adjoint(self, tensor: torch.Tensor, raw_shape: Tuple[int, int] = None) -> torch.Tensor:
        if raw_shape is None:
            raw_shape = self.raw_shape
        if tensor.size(-2) != raw_shape[-2] or tensor.size(-1) != raw_shape[-1]:
            tensor = interpolate(tensor.view(-1, *tensor.shape[-3:]), size=raw_shape[-2:], mode='bilinear', align_corners=False).view(*tensor.shape[:-3], -1, *raw_shape[-2:])
        return tensor

    def forward(self, tensor: torch.Tensor, is_fit: bool = True) -> torch.Tensor:
        if self.is_norm:
            tensor, _, _ = self.norm(tensor, is_fit=is_fit)
        
        if self.is_resize:
            tensor, _ = self.resize(tensor, is_fit=is_fit)

        if self.is_pad:
            tensor, _ = self.pad(tensor, is_fit=is_fit)
        return tensor

    def adjoint(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_pad:
            tensor = self.pad_adjoint(tensor)

        if self.is_resize:
            tensor = self.resize_adjoint(tensor)

        if self.is_norm:
            tensor = self.norm_adjoint(tensor)
        return tensor