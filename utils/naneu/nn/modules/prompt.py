import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

class EncodedPrompt2d(nn.Module):
    """
    https://github.com/hellopipu/PromptMR
    """
    def __init__(self, n_token=5, n_out_feat=128, size=96, n_in_feat=192, learnable=True):
        super().__init__()
        self.ta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_in_feat, n_token, 1, padding=0),
            Rearrange('B T H W -> B T 1 H W'),
            nn.Softmax(dim=1),
        )

        self.param = nn.Parameter(torch.rand(1, n_token, n_out_feat, size, size), 
                                         requires_grad=learnable) # B T C H W
        self.dec = nn.Conv2d(n_out_feat, n_out_feat, kernel_size=3, stride=1, padding=1, bias=False)
                                                                                                                                                                                                                                   
    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        token_weights = self.ta(ref) # B T C H W
        prompt = self.param * token_weights

        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, ref.shape[-2:], mode="bilinear")
        prompt = self.dec(prompt)

        return prompt
    
class EncodedPromptWithFallback2d(nn.Module):
    def __init__(self, n_token=5, n_out_feat=128, size=96, n_in_feat=192, learnable=True):
        super().__init__()
        self.ta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_in_feat, n_token, 1, padding=0),
            Rearrange('B T H W -> B T 1 H W'),
            nn.Softmax(dim=1),
        )

        self.param = nn.Parameter(torch.rand(1, n_token, n_out_feat, size, size), 
                                         requires_grad=learnable) # B T C H W
        self.param_fallback = nn.Parameter(torch.rand(1, n_out_feat, size, size),
                                           requires_grad=learnable) # B T C H W
        self.dec = nn.Conv2d(n_out_feat, n_out_feat, kernel_size=3, stride=1, padding=1, bias=False)

    def normalized_entropy(self, x: torch.Tensor) -> torch.Tensor:
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
                                                                                                                                                                                                                                   
    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        token_weights = self.ta(ref) # B T C H W
        fallback_weights = self.normalized_entropy(token_weights)
        prompt = self.param * token_weights
        prompt = torch.sum(prompt, dim=1) + self.param_fallback * fallback_weights

        prompt = F.interpolate(prompt, ref.shape[-2:], mode="bilinear")
        prompt = self.dec(prompt)

        return prompt