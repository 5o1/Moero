import torch
from torch import nn, SymInt
from torch.nn import functional as F
from typing import Callable, Sequence, Union
from einops import repeat

def conv(in_channels: int, out_channels: int, kernel_size, padding = 'same', bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)

class AverageConvnd(nn.Module):
    kernel: torch.Tensor
    conv_fn: Callable[..., torch.Tensor]

    def __init__(self, size: Sequence[Union[int, SymInt]]):
        super().__init__()
        self.size = size

        ndim = len(size)
        if ndim not in [1, 2, 3]:
            raise ValueError("length of `size` must be 1, 2 or 3.")
        
        self.register_buffer("kernel", self._make_average_kernel(size))
        self.conv_fn = getattr(F, f"conv{ndim}d")
    def _make_average_kernel(self, size: Sequence[Union[int, SymInt]]) -> torch.Tensor:
        kernel = torch.ones(size)
        kernel = kernel / kernel.numel()  # normalize
        kernel = kernel.view(1,1,*kernel.shape)
        return kernel
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, -1, *self.kernel.shape[-2:]).type_as(input)
        input = self.conv_fn(input, kernel, padding =  [s // 2 for s in self.size], groups=in_channels)
        return input

class GaussianConvnd(nn.Module):
    kernel: torch.Tensor
    conv_fn: Callable[..., torch.Tensor]

    def __init__(self, size: Sequence[Union[int, SymInt]], *, sigma: float = 1.5):
        super().__init__()
        self.size = size
        self.sigma = sigma

        ndim = len(size)
        if ndim not in [1, 2, 3]:
            raise ValueError("length of `size` must be 1, 2 or 3.")
        
        self.register_buffer("kernel", self._make_gaussian_kernel(size, sigma = sigma))
        self.conv_fn = getattr(F, f"conv{ndim}d")
        
    def _make_gaussian_kernel(self, size: Sequence[Union[int, SymInt]], *,  sigma: float = 1.5) -> torch.Tensor:
        coords = [torch.arange(size_dim, dtype=torch.float32) - (size_dim - 1) / 2.0 for size_dim in size]
        grids = torch.meshgrid(*coords, indexing="ij")
        kernel = torch.exp(-torch.stack(grids).pow(2).sum(0) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1,1,*kernel.shape)
        return kernel
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, -1, *self.kernel.shape[-2:]).type_as(input)
        input = self.conv_fn(input, kernel, padding =  [s // 2 for s in self.size], groups=in_channels)
        return input

class ChannelAttention2d(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            conv(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weights = self.net(input)
        return input * weights


class CAB2d(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, reduction, bias, act):
        super().__init__()

        self.net = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
            ChannelAttention2d(n_feat, reduction, bias=bias)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = self.net(input)
        input = (input + res) * 0.5
        return input

class CABChain(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_cab, kernel_size, reduction, bias, act, is_res = True):
        super().__init__()
        self.is_res = is_res

        self.encoder = nn.Sequential(
            *[CAB2d(n_in_feat, kernel_size, reduction, bias=bias, act=act) 
              for _ in range(n_cab)]) if n_cab > 0 else nn.Identity()
        self.proj = conv(n_in_feat, n_out_feat, kernel_size, bias=bias) if n_in_feat != n_out_feat else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = self.encoder(input)
        if self.is_res:
            input = (input + res) * 0.5
        else:
            input = res
        input = self.proj(input)
        return input

class MomentumConv2d(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, n_history):
        super().__init__()

        self.n_history = n_history
        self.conv = nn.Sequential(
            conv(n_feat*(n_history+1), n_feat, 1, bias=bias),
            CAB2d(n_feat, kernel_size, reduction, bias=bias, act=act)
        )

    def forward(self, input: torch.Tensor, history: torch.Tensor = None) -> torch.Tensor:
        if history is None:
            input = repeat(input, 'b c h w -> b (c repeat) h w', repeat=self.n_history+1)
        else:
            input = torch.cat([input, history], dim=1)
        input = self.conv(input)
        return input