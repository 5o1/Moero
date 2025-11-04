from typing import List
import torch
from torch import nn
from ..modules.conv import DownBlock, CABChain, UpBlock, PromptUpBlock
from ..modules.prompt import PromptBlock
from utils.naneu.helpers.context import register_extra_output, register_extra_metric
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from utils.naneu.common.importlib import LazyModule


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

class ExpertPool(nn.Module):
    def __init__(
            self, experts: List[nn.Module]
        ):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor, route_weights: torch.Tensor, route_mask: torch.Tensor):
        """
        x : b ref c h w
        route_weights : b poolsize
        route_mask : b poolsize
        """
        x_dense = None
        experts_indices = route_mask.sum(dim=0).nonzero(as_tuple=True)[0]  # Indices of experts that are selected by at least one sample

        for expert_idx in experts_indices:
            expert = self.experts[expert_idx]

            # To sparse the batch
            batch_indices = route_mask[:, expert_idx].nonzero(as_tuple=True)[0] # Indices of samples that select this expert

            x_expert = expert(x[batch_indices])
            # Initialize the dense lists 
            if x_dense is None:
                x_dense = torch.zeros(x.size(0), *x_expert.shape[1:], device=x_expert.device, dtype=x_expert.dtype)

            # Aggregate results to dense lists
            x_dense = x_dense.index_add_(0, batch_indices, x_expert * route_weights[batch_indices, expert_idx].view(-1, *([1]* (x_expert.ndim - 1))))

        return x_dense



class SkipMoeUnet(nn.Module):
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
            moe_poolsize: int,
            moe_top_k: int,
            gate_class_path: str,
            gate_init_args: dict = {},
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
        self.moe_poolsize = moe_poolsize

        # Feature extraction
        self.to_input = nn.Conv2d(in_channels, pyramid_channels[0], kernel_size=kernel_size, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

        # Encoder - 3 DownBlocks
        self.enc = torch.nn.ModuleList([
            DownBlock(pyramid_channels[i], pyramid_channels[i + 1], n_enc_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Gate
        self.gate = LazyModule(gate_class_path)(
            in_channels=pyramid_channels[self.depth],
            poolsize=moe_poolsize,
            idx=idx_cascade,
            top_k=moe_top_k,
            **gate_init_args
        )

        # Skip Connections - 3 SkipBlocks
        self.skip = torch.nn.ModuleList([
            ExpertPool([
                SkipBlock(pyramid_channels[i], n_skip_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias) for _ in range(moe_poolsize)
            ]) for i in range(self.depth)
        ])

        # Bottleneck
        self.skip_bottleneck = ExpertPool([
            self.skip_bottleneck for _ in range(moe_poolsize)
        ])

        # Decoder - 3 UpBlocks
        self.prompt = torch.nn.ModuleList([
            PromptBlock(pyramid_channels[i + 1], prompt_channels[i], prompt_tokens[i], prompt_figsize[i], bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        self.dec = torch.nn.ModuleList([
            PromptUpBlock(pyramid_channels[i + 1], pyramid_channels[i], prompt_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, self.n_history, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
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

        # 2. moe + gate
        route_weights, route_mask, route_idx = self.gate(x.mean(1))  # x: b ref c h w -> b c h w -> b
        experts_indices = route_mask.sum(dim=0).nonzero(as_tuple=True)[0]  # Indices of experts that are selected by at least one sample

        x_dense = None

        for expert_idx in experts_indices:
            expert = self.dec[expert_idx]

            # To sparse the batch
            batch_indices = route_mask[:, expert_idx].nonzero(as_tuple=True)[0] # Indices of samples that select this expert

            x_expert = expert(
                x[batch_indices],
                [residual[ilevel][batch_indices] for ilevel in range(len(residual))]
            )
            # Initialize the dense lists 
            if x_dense is None:
                x_dense = torch.zeros(x.size(0), *x_expert.shape[1:], device=x_expert.device, dtype=x_expert.dtype)

            # Aggregate results to dense lists
            x_dense = x_dense.index_add_(0, batch_indices, x_expert * route_weights[batch_indices, expert_idx].view(-1, *([1]* (x_expert.ndim - 1))))

        x = x_dense

        return x
