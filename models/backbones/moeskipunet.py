from typing import List
import torch
import torch.distributed as dist
from torch import nn
from ..modules.conv import DownBlock, CABChain, UpBlock, PromptUpBlock, CAB
from ..modules.prompt import PromptBlock
from utils.naneu.helpers.context import register_extra_output, register_extra_metric
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from utils.naneu.common.importlib import LazyModule
from ..modules.nav import RouterOp


def sparse_forward(module: nn.Module, x: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dense = None
    for expert_idx in indices:
        expert = module[expert_idx]

        # To sparse the batch
        batch_indices = mask[:, expert_idx].nonzero(as_tuple=True)[0] # Indices of samples that select this expert
        x_expert = expert(
            x[batch_indices],
        )

        # Initialize the dense lists
        if dense is None:
            dense = torch.zeros(x.size(0), *x_expert.shape[1:], device=x_expert.device, dtype=x_expert.dtype)

        # Aggregate results to dense lists
        dense = dense.index_add_(0, batch_indices, x_expert * weights[batch_indices, expert_idx].view(-1, *([1]* (x_expert.ndim - 1))))
    return dense

@torch.no_grad()
def sync(src_module: torch.nn.Module, tgt_modules: List[torch.nn.Module]):
    src_sd = src_module.state_dict()
    for i, tgt in enumerate(tgt_modules):
        tgt.load_state_dict(src_sd, strict=True)

@torch.no_grad()
def sync_ddp(src_module: torch.nn.Module, tgt_modules: List[torch.nn.Module], process_group=None):
    if not dist.is_available() or not dist.is_initialized():
        sync(src_module, tgt_modules)
        print("DDP not initialized, only sync within single process.")
        return

    world_size = dist.get_world_size(group=process_group)
    if world_size < 2:
        sync(src_module, tgt_modules)
        print("Only one process in the group, only sync within single process.")
        return

    src_rank = 0

    for p in src_module.parameters():
        dist.broadcast(p, src=src_rank, group=process_group)
    for b in src_module.buffers():
        dist.broadcast(b, src=src_rank, group=process_group)

    print(f"Broadcasted source module from rank {src_rank}.")

    sync(src_module, tgt_modules)

    print("Synchronized all target modules across processes.")

    dist.barrier(group=process_group)

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

class ExpertSkipGroup(nn.Module):
    def __init__(self, in_channels: List[int], n_skip_cab: List[int], moe_poolsize: int, kernel_size: int, reduction: int, dropout: float, bias: bool = True, norm: bool = False):
        super().__init__()
        self.moe_poolsize = moe_poolsize
        self.skip = torch.nn.ModuleList([
            torch.nn.ModuleList([
                SkipBlock(in_channels[i], n_skip_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias)
                for _ in range(moe_poolsize)
            ])
            for i in range(len(in_channels) - 1)
        ])

        self.skip.append(
            torch.nn.ModuleList([
                BottleNeck(in_channels[-1], n_skip_cab[-1], kernel_size, reduction, dropout, norm=norm, bias=bias)
                for _ in range(moe_poolsize)
            ])
        )

        self.demodem = torch.nn.ModuleList([
            CAB(in_channels[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(len(in_channels))
        ])


    @torch.no_grad()
    def sync(self, src_expert_indice: List[int]):
        for level in range(len(self.skip)):
            sync_ddp(
                self.skip[level][src_expert_indice[level]],
                [self.skip[level][i] for i in range(self.moe_poolsize) if i != src_expert_indice[level]]
            )

    def forward(self, residual: List[torch.Tensor], weights: List[torch.Tensor], mask: List[torch.Tensor], indices: List[torch.Tensor]) -> torch.Tensor:
        results = [None for _ in range(len(residual))]

        # Skip connections
        for level in range(len(residual)):
            results[level] = sparse_forward(
                self.skip[level],
                residual[level],
                weights[level],
                mask[level],
                indices[level]
            )

            results[level] = self.demodem[level](results[level])

        return results
        

class MoeSkipUnet(nn.Module):
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
            n_history: int = 0,
            bias: bool = True,
            norm: bool = False,
        ):
        if n_history > 0 and moe_top_k > 1:
            raise ValueError("Top-k > 1 with history is not supported yet.")

        self.depth = len(pyramid_channels) - 1
        self.n_history = n_history
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
        self.to_input = nn.Conv2d(in_channels, pyramid_channels[0], kernel_size=3, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

        # Encoder - 3 DownBlocks
        self.enc = torch.nn.ModuleList([
            DownBlock(pyramid_channels[i], pyramid_channels[i + 1], n_enc_cab[i], kernel_size, reduction, dropout, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # Gate
        self.gate = LazyModule(gate_class_path)(
            in_channels=pyramid_channels,
            poolsize=moe_poolsize,
            idx=idx_cascade,
            top_k=moe_top_k,
            **gate_init_args
        )

        self.skip_group = ExpertSkipGroup(
            in_channels=pyramid_channels,
            n_skip_cab=n_skip_cab,
            moe_poolsize=moe_poolsize,
            kernel_size=kernel_size,
            reduction=reduction,
            dropout=dropout,
            bias=bias,
            norm=norm,
        )

        # Decoder - 3 UpBlocks
        self.prompt = torch.nn.ModuleList([
            PromptBlock(pyramid_channels[i + 1], prompt_channels[i], prompt_tokens[i], prompt_figsize[i], bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])
        self.dec = torch.nn.ModuleList([
            PromptUpBlock(pyramid_channels[i + 1], pyramid_channels[i], prompt_channels[i], n_dec_cab[i], kernel_size, reduction, dropout, n_history=n_history, norm=norm, bias=bias).rearrange("b ref c h w -> (b ref) c h w")
            for i in range(self.depth)
        ])

        # OutConv
        self.to_output = nn.Conv2d(pyramid_channels[0], out_channels, kernel_size=5, padding="same", bias=bias).rearrange("b ref c h w -> (b ref) c h w")

        # # freeze for transfer learning
        for module in [self.to_input, self.enc, self.prompt, self.dec, self.to_output]:
            module.requires_grad_(False)

    def forward(self, x: torch.Tensor, history: List[List[torch.Tensor]] | None = None):
        """
        Real. Complex dimension have bound to channel dimension.
        x : b ref c h w
        """
        if history is None or not self.n_history > 0:
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

        cache = [None for _ in range(self.depth)]
        residual = [None for _ in range(self.depth + 1)]

        # 0. featue extraction
        x = self.to_input(x)

        # 1. encoder
        for i in range(self.depth):
            x, residual[i] = self.enc[i](x)

        residual[self.depth] = x

        # 2. moe + gate
        route_inputs = [residual_level.mean(1) for residual_level in residual]  # b ref c h w -> b c h w
        route_weights, route_mask, route_idx, ops = self.gate(*route_inputs)  # x: b ref c h w -> b c h w -> b
        experts_indices = [route_mask[level].sum(dim=0).nonzero(as_tuple=True)[0] for level in range(route_mask.size(0))]

        # Response for ops
        for key, item in ops.items():
            if key == RouterOp.Sync:
                # Sync parameters for all experts
                print(f"Syncing experts at main expert idx {item}.")
                self.skip_group.sync(item)

        residual = self.skip_group(
            residual,
            route_weights,
            route_mask,
            experts_indices
        )

        x = residual[self.depth]
        residual = residual[:self.depth]

        # 3. decoder
        for i in range(self.depth - 1, -1, -1):
            cache[i] = x
            x = self.dec[i](x, self.prompt[i](x), residual[i], history[i])

        x = self.to_output(x)

        if self.n_history > 0:
            return x, cache
        else:
            return x