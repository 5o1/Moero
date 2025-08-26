# import torch
# from torch import nn

# from ..modules.attention import MultiheadSelfAttention

# class MlpFeedForward(nn.Module):
#     def __init__(self, n_in_feat: int, n_hidden_feat: int, dropout: float = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_in_feat, n_hidden_feat),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(n_hidden_feat, n_in_feat),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# class TransformerBlock(nn.Module):
#     def __init__(self, n_feat: int, n_head: int, n_head_feat: int, n_mlp_feat: int, dropout: float = 0.):
#         super().__init__()
#         self.norm = nn.LayerNorm(n_feat)
#         self.attn = MultiheadSelfAttention(n_feat, n_head = n_head, n_head_feat = n_head_feat, dropout = dropout),
#         self.ff = MlpFeedForward(n_feat, n_mlp_feat, dropout = dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.attn(self.norm(x)) + x
#         x = self.ff(self.norm(x)) + x