import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .conv import ConvBlock, CABChain, ResChain
from .vqvae import Quantize

def normalized_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized entropy of a tensor.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized entropy of the input tensor.
    """
    entropy = -torch.sum(x * torch.log(x + 1e-12), dim=1, keepdim=True)
    num_classes = x.size(1)
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

class Prompt2d(nn.Module):
    # Source: https://github.com/hellopipu/PromptMR-plus
    def __init__(self, in_channels: int, out_channels: int, n_tokens: int, figsize: int = 96):
        super().__init__()
        self.proj = nn.Linear(in_channels, n_tokens)
        self.prompt_param = nn.Parameter(torch.rand(1, n_tokens, out_channels, figsize, figsize), requires_grad=False) # B seq_len seq_feat h w
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : B C H W
        """
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.proj(emb), dim=1) # B seq_len
        prompt_param = self.prompt_param.repeat(B, *([1] * (self.prompt_param.ndim - 1)))
        prompt = rearrange(prompt_weights, "b seq_len -> b seq_len 1 1 1") * prompt_param
        prompt = torch.sum(prompt, dim=1) # b seq_feat h w

        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=True)

        return prompt

class PromptBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_tokens: int = 4,
        figsize: int = 128,
        *,
        bias: bool = False
    ):
        super().__init__()
        self.promptlayer = Prompt2d(in_channels, out_channels, n_tokens, figsize)
        self.dec = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = "same", bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input : B C H W
        """
        x = self.promptlayer(x) # B C H W
        x = self.dec(x)
        return x

    
class VQPromptBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_tokens: int = 512,
        n_res_block: int = 3,
        reduction: int = 4,
        decay: float = 0.99,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.enc = nn.Identity()
        self.quantize = Quantize(in_channels, n_tokens, decay=decay)
        self.dec = nn.Sequential(
            ResChain(in_channels, n_res_block, reduction),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def bincount_batch(self, tensor: torch.Tensor, bin_length: int) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError("bincount_batch input must be 2D.")
        if bin_length < (tensor.max() - tensor.min()):
            raise ValueError("bin_length must be greater than the range of input tensor values.")
        bin = torch.zeros(tensor.size(0), bin_length, device=tensor.device, dtype=torch.int64)
        bin.scatter_add_(dim = 1, index = tensor, src = torch.ones_like(tensor, dtype=torch.int64))
        return bin

    def forward(self, input: torch.Tensor):
        quant = self.enc(input).permute(0, 2, 3, 1) # B H W C
        quant, diff, idx = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2) # B C H W
        quant = self.dec(quant)
        diff = diff.unsqueeze(0)

        # Word frequency count
        # id: B H W
        with torch.no_grad():
            wordfreq = self.bincount_batch(idx.view(idx.size(0), -1), bin_length=self.n_tokens).float()
            pixel_count = idx[0].numel()
            wordfreq = wordfreq / pixel_count # B n_tokens

        return quant, diff, wordfreq

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)

        dec = self.dec(quant)
        return dec