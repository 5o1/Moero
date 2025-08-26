from torch import nn
import torch
from typing import Optional, List
import math

class ConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0, stride: int = 1, padding: str = 'same',
            *,
            norm: bool = False, bias: bool = True
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True, eps = 1e-8) if norm else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = True if norm else bias),
            nn.PReLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool = False, bias: bool = True
        ):
        super().__init__()
        self.encoder = CABChain(in_channels, n_cab, kernel_size, reduction, dropout, norm = norm, bias = bias)
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias = True)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, prompt_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float, n_history: int = 0,
            *,
            norm: bool = False, bias: bool = True,
            history_norm:bool = False
        ):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        self.is_history_norm = history_norm

        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_channels * (n_history + 1), in_channels, kernel_size=1, padding="same", bias = bias),
                CAB(in_channels, kernel_size, reduction, dropout, norm = norm, bias = bias)
            )

        self.fuse = nn.Sequential(
            CABChain(in_channels+prompt_channels, n_cab, kernel_size, reduction, dropout, norm = norm, bias = bias)
        )
        self.reduce = nn.Conv2d(in_channels + prompt_channels, in_channels, kernel_size=1, bias = bias)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = bias)
            )

        self.to_out = CAB(out_channels, kernel_size, reduction, dropout, norm = norm, bias = bias)

        # norm
        if n_history > 0 and history_norm:
            self.history_norm = nn.InstanceNorm2d(in_channels * (n_history), affine=True)

    def forward(self, x: torch.Tensor, prompt_dec: torch.Tensor, skip: torch.Tensor, history: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.n_history > 0:
            if history is None:
                x = torch.tile(x, (1, self.n_history+1, 1, 1))
            elif (history.size(-3) % self.n_history != 0) or (history.size(-3) // self.n_history != x.size(-3)):
                raise ValueError(f"Unexpected history size: {history.size(-3)}. Expected to be a multiple of {self.n_history} times the channels of x: {x.size(-3)}.")
            else:
                if self.is_history_norm:
                    history = self.history_norm(history)
                x = torch.cat([x, history], dim=-3)
            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)
        x = self.up(x)
        x = x + skip
        x *= 0.5
        x = self.to_out(x)
        return x


class CALayer(nn.Module):
    def __init__(
            self, channels: int, reduction: int,
            *,
            bias: bool = True
        ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_up = nn.Sequential(
            nn.Conv2d(channels, math.ceil(channels / reduction), 1, padding=0, bias = bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(channels / reduction), channels, 1, padding=0, bias = bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.down_up(y)
        return x * y


class CAB(nn.Module):
    def __init__(
            self, in_channels: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool = True, bias: bool = True
        ):
        super().__init__()
        self.ca = CALayer(in_channels, 4, bias = bias)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, math.ceil(in_channels / reduction), kernel_size=kernel_size, dropout=dropout, norm = norm, bias = bias),
            nn.Conv2d(math.ceil(in_channels / reduction), in_channels, kernel_size=kernel_size, padding="same", bias = bias)
        )
        
    def forward(self, x):
        res = self.conv(x)
        res = self.ca(res)
        res += x
        res *= 0.5
        return res


class CABChain(nn.Module):
    def __init__(
            self, in_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool = True, bias: bool = True
        ):
        super().__init__()
        if n_cab == 0:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Sequential(*[
                CAB(in_channels, kernel_size, reduction, dropout, norm = norm, bias = bias)
                for i in range(n_cab)
                ])

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        res = self.layer(x)
        return res


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, math.ceil(in_channels / reduction), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(in_channels / reduction), in_channels, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        out += input
        out *= 0.5
        return out


class ResChain(nn.Module):
    def __init__(self, in_channels: int, n_res_block: int, reduction: int):
        super().__init__()
        if n_res_block == 0:
            self.layer = nn.Identity()
        else:
            blocks = [nn.Conv2d(in_channels, in_channels, 3, padding=1)]
            blocks += [ResBlock(in_channels, reduction) for _ in range(n_res_block)]
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.layer(x)
        return res