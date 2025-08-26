from typing import List, Optional
import torch
from torch import nn
from .modules.conv import DownBlock, CABChain, UpBlock, CAB
from .modules.prompt import PromptBlock, VQPromptBlock
from utils.naneu.helpers.context import register_extra_output, register_extra_loss
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from einops import rearrange


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
            self, in_channels: int, prompt_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool, bias: bool   
        ):
        super().__init__()
        self.fuse = CABChain(in_channels+prompt_channels, n_cab, kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
        self.reduce = nn.Conv2d(in_channels + prompt_channels, in_channels, kernel_size=1, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
        self.decoder : CAB = CAB(in_channels, kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x , prompt], dim=-3)
        x = self.fuse(x)
        x = self.reduce(x)
        x = self.decoder(x)
        return x


class PromptVQUnet(nn.Module):
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
            idx_cascade = None,
            n_history: int = None,
            bias: bool = True,
            norm: bool = False,
            history_norm: bool = False,
        ):
        self.depth = len(pyramid_channels) - 1
        if not all([
            len(pyramid_channels) == self.depth + 1,
            len(prompt_tokens) == self.depth + 1,
            len(prompt_channels) == self.depth + 1,
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
        self.pyramid_channels = pyramid_channels
        self.idx_cascade = idx_cascade
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
        self.prompt_bottleneck = VQPromptBlock(pyramid_channels[self.depth], prompt_channels[self.depth], prompt_tokens[self.depth], 3, reduction, 0.99).rearrange("b ref c h w -> (b ref) c h w", for_output = [0])
        self.skip_bottleneck = BottleNeck(pyramid_channels[self.depth], prompt_channels[self.depth], n_skip_cab[self.depth], kernel_size, reduction, dropout, norm=norm, bias=bias)

        # Decoder - 3 UpBlocks
        self.prompt = torch.nn.ModuleList([
            PromptBlock(pyramid_channels[i + 1], prompt_channels[i], prompt_tokens[i], prompt_figsize[i], bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        self.dec = torch.nn.ModuleList([
            UpBlock(pyramid_channels[i + 1], pyramid_channels[i], prompt_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, self.n_history, norm=norm, bias=bias, history_norm=history_norm).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # OutConv
        self.to_output = nn.Conv2d(pyramid_channels[0], out_channels, 5, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

    def forward(self, x: torch.Tensor, history: Optional[List[List[torch.Tensor]]] = None):
        """
        Real. Complex dimension have bound to channel dimension.
        x : b ref c h w
        """
        if history is None:
            history = [None for _ in range(len(self.pyramid_channels) - 1)]
        else:
            n_cached = len(history[0])
            if not all(n_cached == len(history[d]) for d in range(1, len(history), 1)):
                raise ValueError(f"History must be a list of lists with the same length. Got {[len(h) for h in history]}.")
            if n_cached == 0: # Initialization
                history = [None for _ in range(len(self.pyramid_channels) - 1)]
            elif n_cached < self.n_history: # Padding by first history
                history = [torch.cat(h[:1] * (self.n_history - n_cached) + h, dim=-3)  for h in history]
            else: # Use last self.n_history history
                history = [torch.cat(h[-self.n_history:], dim=-3) for h in history]

        cache = [None for _ in range(len(self.pyramid_channels)-1)]
        residual = [None for _ in range(len(self.pyramid_channels)-1)]

        # register_extra_output(self, f"to_input.{str(self.idx_cascade)}", x.detach().clone())

        # 0. featue extraction
        x = self.to_input(x)

        # 1. encoder
        for i in range(self.depth):
            x, residual[i] = self.enc[i](x)
 
        # 2. bottleneck
        vq, loss, wordfreq = self.prompt_bottleneck(x)
        x = self.skip_bottleneck(x, vq)
        wordfreq = rearrange(wordfreq, "(b ref) words -> b ref words", b = x.size(0), ref = x.size(1)).mean(1)

        # 3. decoder
        for i in range(self.depth - 1, -1, -1):
            cache[i] = x.clone()
            x = self.dec[i](x, self.prompt[i](x), self.skip[i](residual[i]), history[i])

        x = self.to_output(x)

        # Record vq loss
        register_extra_loss(self, f"vq_loss{self.idx_cascade if self.idx_cascade is not None else ''}", loss.mean())

        return x, cache, wordfreq
    
    def get_vq_module(self) -> torch.nn.Module:
        """
        Returns the VQ module from the bottleneck.
        """
        return self.prompt_bottleneck
    
    def set_vq_module(self, vq_module: torch.nn.Module):
        """
        Sets the VQ module in the bottleneck.
        """
        self.prompt_bottleneck = vq_module