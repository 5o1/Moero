from typing import List
import torch
from torch import nn
from ..modules.conv import DownBlock, CABChain, UpBlock
from utils.naneu.helpers.context import register_extra_output, register_extra_metric
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import


class SkipBlock(nn.Module):
    def __init__(
            self, in_channels: int, n_cab:int, kernel_size:int, reduction: int, dropout: float,
            *,
            norm: bool, bias: bool
        ):
        super().__init__()
        self.decoder: CABChain = CABChain(in_channels, n_cab, kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor):
        """
        x : b ref c h w
        """
        x = self.decoder(x)
        return x

class BottleNeck(nn.Module):
    def __init__(
            self, in_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool, bias: bool   
        ):
        super().__init__()
        self.decoder : CABChain = CABChain(in_channels, n_cab, kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x


class Unet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pyramid_channels : List[int], # depth + 1
            n_enc_cab: List[int], # depth
            n_dec_cab: List[int], # depth
            n_skip_cab: List[int], # depth + 1
            kernel_size: int = 3,
            reduction: float | int = 2,
            dropout: float = 0.0,
            decoder_expand: int = 0,
            idx_cascade: int = None,
            bias: bool = True,
            norm: bool = False,
        ):
        self.depth = len(pyramid_channels) - 1
        if not all([
            len(pyramid_channels) == self.depth + 1,
            len(n_dec_cab) == self.depth,
            len(n_skip_cab) == self.depth + 1,
        ]):
            raise ValueError(
                "The length of pyramid_channels,n_enc_cab, n_dec_cab, and n_skip_cab must match the depth of the network. "
                f"Got {pyramid_channels}, {n_enc_cab}, {n_dec_cab}, and {n_skip_cab} respectively."
            )

        super().__init__()
        self.idx_cascade = idx_cascade

        # Feature extraction
        self.to_input = nn.Conv2d(in_channels, pyramid_channels[0], kernel_size=kernel_size, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

        # Encoder - 3 DownBlocks
        self.enc = torch.nn.ModuleList([
            DownBlock(pyramid_channels[i], pyramid_channels[i + 1], n_enc_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Skip Connections - 3 SkipBlocks
        self.skip = torch.nn.ModuleList([
            SkipBlock(pyramid_channels[i], n_skip_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias)
            for i in range(self.depth)
        ])

        # Bottleneck
        self.skip_bottleneck = BottleNeck(pyramid_channels[self.depth], n_skip_cab[self.depth], kernel_size, reduction, dropout, norm=norm, bias=bias)

        # Decoder - 3 UpBlocks
        self.dec = torch.nn.ModuleList([
            UpBlock(pyramid_channels[i + 1], pyramid_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, n_history=decoder_expand, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # OutConv
        self.to_output = nn.Conv2d(pyramid_channels[0], out_channels, 5, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor):
        """
        Real. Complex dimension have bound to channel dimension.
        x : b ref c h w
        """
        residual = [None for _ in range(self.depth)]

        # 0. featue extraction
        x = self.to_input(x)

        # 1. encoder
        for i in range(self.depth):
            x, residual[i] = self.enc[i](x)
 
        # 2. bottleneck
        x = self.skip_bottleneck(x)

        # 3. decoder
        for i in range(self.depth - 1, -1, -1):
            x = self.dec[i](x, self.skip[i](residual[i]))

        x = self.to_output(x)
        return x