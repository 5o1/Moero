from typing import List, Optional, Literal
import torch
import math
from functools import partial
from torch import nn
from ..modules.conv import DownBlock, CABChain, PromptUpBlock, SelfFiLM
from ..modules.prompt import PromptBlock
from utils.naneu.helpers.context import register_extra_output
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


class PromptUnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pyramid_channels : List[int], # depth + 1
            prompt_tokens: List[int], # depth
            prompt_channels: List[int], # depth
            prompt_figsize: List[int], # depth
            n_enc_cab: List[int], # depth
            n_dec_cab: List[int], # depth
            n_skip_cab: List[int], # depth + 1
            kernel_size: int = 3,
            reduction: float | int = 2,
            dropout: float = 0.0,
            idx_cascade: int = None,
            n_cascade: int | None = None,
            n_history: int = 0,
            *,
            bias: bool = True,
            norm: bool = False,
            use_history: bool = True,
            history_norm: Literal["norm", "film"] | None = None,
        ):
        self.depth = len(pyramid_channels) - 1
        if not all([
            len(pyramid_channels) == self.depth + 1,
            len(prompt_tokens) == self.depth,
            len(prompt_channels) == self.depth,
            len(prompt_figsize) == self.depth,
            len(n_enc_cab) == self.depth,
            len(n_dec_cab) == self.depth,
            len(n_skip_cab) == self.depth + 1,
        ]):
            raise ValueError(
                "The length of pyramid_channels, prompt_tokens, prompt_channels, prompt_figsize, n_enc_cab, n_dec_cab, and n_skip_cab must match the depth of the network. "
                f"Got {pyramid_channels}, {prompt_tokens}, {prompt_channels}, {prompt_figsize}, {n_enc_cab}, {n_dec_cab}, and {n_skip_cab} respectively."
            )

        super().__init__()
        self.idx_cascade = idx_cascade
        self.use_history = use_history
        self.history_norm = history_norm

        self.is_first = False
        self.is_last = False
        if idx_cascade is not None and n_cascade is not None:
            self.is_first = (idx_cascade == 0)
            self.is_last = (idx_cascade == n_cascade - 1)

        if n_history is None:
            if idx_cascade is None:
                self.n_history = 0
            else:
                self.n_history = idx_cascade
        else:
            self.n_history = n_history

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
        self.prompt = torch.nn.ModuleList([
            PromptBlock(pyramid_channels[i + 1], prompt_channels[i], prompt_tokens[i], prompt_figsize[i], bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        self.dec = torch.nn.ModuleList([
            PromptUpBlock(pyramid_channels[i + 1], pyramid_channels[i], prompt_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, self.n_history, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Feature cache adapters
        if history_norm and use_history:
            if not self.is_last:
                self.cache_to_output = nn.ModuleList([
                    nn.Conv2d(pyramid_channels[i + 1], math.ceil(pyramid_channels[i + 1] // 2), kernel_size=1, bias=True).rearrange("b ref c h w -> (b ref) c h w")
                    for i in range(self.depth)
                ])
            if not self.is_first:
                if history_norm == "norm":
                    norm_layer = partial(nn.InstanceNorm2d, affine=True)
                elif history_norm == "film":
                    norm_layer = partial(SelfFiLM, bias = True)
                else:
                    raise ValueError(f"Unsupported history_norm: {history_norm}. Supported values are 'norm' and 'film'.")
                self.cache_to_input = nn.ModuleList([
                    nn.Sequential(
                        norm_layer(math.ceil(pyramid_channels[i + 1] // 2) * self.n_history),
                        nn.Conv2d(math.ceil(pyramid_channels[i + 1] // 2) * self.n_history, pyramid_channels[i + 1] * self.n_history, kernel_size=1, bias=True)
                    ).rearrange("b ref c h w -> (b ref) c h w")
                    for i in range(self.depth)
                ])

        # OutConv
        self.to_output = nn.Conv2d(pyramid_channels[0], out_channels, 5, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor, history: Optional[List[List[torch.Tensor]]] = None):
        """
        Real. Complex dimension have bound to channel dimension.
        x : b ref c h w
        """
        if history is None or not self.use_history:
            history = [None for _ in range(self.depth)]
        else:
            n_cached = len(history[0])
            if not all(n_cached == len(history[d]) for d in range(1, len(history), 1)):
                raise ValueError(f"History must be a list of lists with the same length. Got {[len(h) for h in history]}.")
            if n_cached == 0: # Initialization
                history = [None for _ in range(self.depth)]
            else:
                if n_cached < self.n_history: # Padding by first history
                    history = [torch.cat(h[:1] * (self.n_history - n_cached) + h, dim=-3)  for h in history]
                else: # Use last self.n_history history
                    history = [torch.cat(h[-self.n_history:], dim=-3) for h in history]
                # History norm
                if self.history_norm and not self.is_first:
                    history = [self.cache_to_input[i](history[i]) for i in range(self.depth)]

        cache = [None for _ in range(self.depth)]
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
            cache[i] = self.cache_to_output[i](x) if self.history_norm and self.use_history and not self.is_last else x
            x = self.dec[i](x, self.prompt[i](x), self.skip[i](residual[i]), history[i])

        x = self.to_output(x)
        return x, cache