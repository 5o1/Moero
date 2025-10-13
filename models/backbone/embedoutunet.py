from typing import List, Optional
import torch
from torch import nn
from ..modules.conv import DownBlock, CABChain, PromptUpBlock, CAB, UpBlock
from ..modules.prompt import PromptBlock, VQPromptBlock
from utils.naneu.helpers.context import register_extra_output, register_extra_loss
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from einops import rearrange
from einops.layers.torch import Rearrange

class EmbedModule(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def prompt_channels(self) -> int:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @property
    def embedding_channels(self) -> int:
        raise NotImplementedError("This method should be implemented by subclasses.")


class VQEmbedModule(EmbedModule):
    def __init__(
            self,
            in_channels: int,
            prompt_channels: int,
            embedding_channels: int,
            n_conv: int = 3,
            reduction: int = 4,
            decay: float = 0.99,
        ):
        super().__init__()
        self.prompt_channels = prompt_channels
        self.embedding_channels = embedding_channels
        self.vqblock = VQPromptBlock(in_channels, prompt_channels, embedding_channels, n_conv, reduction, decay).rearrange("b ref c h w -> (b ref) c h w", for_output = [0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : b ref c h w
        """
        prompt, loss, embedding = self.vqblock(x)
        embedding = rearrange(embedding, "(b ref) words -> b ref words", b = x.size(0), ref = x.size(1)).mean(1)
        return prompt, embedding, loss
    

class ConvEmbedModule(EmbedModule):
    def __init__(
            self,
            in_channels: int,
            prompt_channels: int,
            embedding_channels: int,
            n_conv:int = 3,
            kernel_size: int = 3,
        ):
        super().__init__()
        self.prompt_channels = prompt_channels
        self.embedding_channels = embedding_channels
        self.conv = nn.Conv2d(in_channels, prompt_channels, kernel_size=kernel_size, padding="same", bias=False).rearrange("b ref c h w -> (b ref) c h w")
        _layers = []
        for i in range(n_conv):
            _layers.extend([
                nn.InstanceNorm2d(prompt_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(prompt_channels, embedding_channels if i == n_conv-1 else prompt_channels, kernel_size=kernel_size, padding="same", bias=False),
            ])
        self.embed = nn.Sequential(*_layers).rearrange("b ref c h w -> (b ref) c h w")
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c")
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        x : b ref c h w
        """
        prompt = self.conv(x)
        embedding = self.embed(prompt).mean(dim=1)
        embedding = self.pool(embedding)
        return prompt, embedding, None


class SkipBlock(nn.Module):
    def __init__(
            self, in_channels: int, n_cab:int, kernel_size:int, reduction: int, dropout: float,
            *,
            norm: bool, bias: bool
        ):
        super().__init__()
        self.decoder: CABChain = CABChain(in_channels, n_cab, kernel_size, reduction, dropout, norm=norm, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.decoder(x)
        return x


class BottleNeck(nn.Module):
    def __init__(
            self, in_channels: int, embed_feat_channels: int, n_cab: int, kernel_size: int, reduction: int, dropout: float,
            *,
            norm: bool, bias: bool   
        ):
        super().__init__()
        self.fuse = CABChain(in_channels+embed_feat_channels, n_cab, kernel_size, reduction, dropout, norm=norm, bias=bias)
        self.reduce = nn.Conv2d(in_channels + embed_feat_channels, in_channels, kernel_size=1, bias=bias)
        self.decoder : CAB = CAB(in_channels, kernel_size, reduction, dropout, norm=norm, bias=bias)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x , prompt], dim=-3)
        x = self.fuse(x)
        x = self.reduce(x)
        x = self.decoder(x)
        return x


class EmbedOutUnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pyramid_channels : List[int], # depth + 1
            prompt_tokens: List[int] | None, # depth or None
            prompt_channels: List[int] | None, # depth or None
            prompt_figsize: List[int] | None, # depth or None
            n_enc_cab: List[int], # depth
            n_dec_cab: List[int], # depth
            n_skip_cab: List[int], # depth + 1
            kernel_size: int = 3,
            reduction: float | int = 2,
            dropout: float = 0.0,
            idx_cascade = None,
            n_history: int | None = None,
            embed_module: EmbedModule = None,
            *,
            bias: bool = True,
            norm: bool = False,
            history_norm: bool = False,
        ):
        self.depth = len(pyramid_channels) - 1
        self.use_figprompt = True if prompt_tokens is not None and prompt_channels is not None and prompt_figsize is not None else False

        if self.use_figprompt and not all([
            len(prompt_tokens) == self.depth,
            len(prompt_channels) == self.depth,
            len(prompt_figsize) == self.depth,
        ]):
            raise ValueError(
                "The length of prompt_tokens, prompt_channels, and prompt_figsize must match the depth of the network when using figprompt. "
                f"Got {prompt_tokens}, {prompt_channels}, and {prompt_figsize} respectively."
            )

        if not all([
            len(n_enc_cab) == self.depth,
            len(n_dec_cab) == self.depth,
            len(n_skip_cab) == self.depth + 1,
        ]):
            raise ValueError(
                "The length of  n_enc_cab, n_dec_cab, and n_skip_cab must match the depth of the network. "
                f"Got  {n_enc_cab}, {n_dec_cab}, and {n_skip_cab} respectively."
            )

        super().__init__()
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

        # Encoder - DownBlocks
        self.enc = torch.nn.ModuleList([
            DownBlock(pyramid_channels[i], pyramid_channels[i + 1], n_enc_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Skip Connections - SkipBlocks
        self.skip = torch.nn.ModuleList([
            SkipBlock(pyramid_channels[i], n_skip_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Bottleneck
        self._embed_module = embed_module
        self.bottleneck = BottleNeck(pyramid_channels[self.depth], self._embed_module.prompt_channels, n_skip_cab[self.depth], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")

        # Decoder - UpBlocks
        if self.use_figprompt:
            self.prompt = torch.nn.ModuleList([
                PromptBlock(pyramid_channels[i + 1], prompt_channels[i], prompt_tokens[i], prompt_figsize[i], bias=bias).rearrange("b ref c h w -> (b ref) c h w")
                for i in range(self.depth)
            ])
            self.dec = torch.nn.ModuleList([
                PromptUpBlock(pyramid_channels[i + 1], pyramid_channels[i], prompt_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, self.n_history, norm=norm, bias=bias, history_norm=history_norm).rearrange("b ref c h w -> (b ref) c h w")
                for i in range(self.depth)
            ])
        else:
            self.dec = torch.nn.ModuleList([
                UpBlock(pyramid_channels[i + 1], pyramid_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, self.n_history, norm=norm, bias=bias, history_norm=history_norm).rearrange("b ref c h w -> (b ref) c h w")
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
            history = [None for _ in range(self.depth)]
        else:
            n_cached = len(history[0])
            if not all(n_cached == len(history[d]) for d in range(1, len(history), 1)):
                raise ValueError(f"History must be a list of lists with the same length. Got {[len(h) for h in history]}.")
            if n_cached == 0: # Initialization
                history = [None for _ in range(self.depth)]
            elif n_cached < self.n_history: # Padding by first history
                history = [torch.cat(h[:1] * (self.n_history - n_cached) + h, dim=-3)  for h in history]
            else: # Use last self.n_history history
                history = [torch.cat(h[-self.n_history:], dim=-3) for h in history]

        cache = [None for _ in range(self.depth)]
        residual = [None for _ in range(self.depth)]

        # 0. featue extraction
        x = self.to_input(x)

        # 1. encoder
        for i in range(self.depth):
            x, residual[i] = self.enc[i](x)

        # 2. bottleneck
        embed_feat, embed_out, loss = self._embed_module(x)
        x = self.bottleneck(x, embed_feat)

        # 3. decoder
        for i in range(self.depth - 1, -1, -1):
            cache[i] = x.clone()
            x = self.dec[i](x, self.prompt[i](x), self.skip[i](residual[i]), history[i])

        x = self.to_output(x)

        # Register extra outputs and losses
        if loss is not None:
            register_extra_loss(self, f"embed_loss{self.idx_cascade if self.idx_cascade is not None else ''}", loss.mean())
        return x, cache, embed_out

    @property
    def embed_module(self) -> EmbedModule:
        return self._embed_module

    @embed_module.setter
    def embed_module(self, module: EmbedModule):
        self._embed_module = module