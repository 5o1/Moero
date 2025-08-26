import torch
from torch import nn
from einops import rearrange


class MultiheadSelfAttention(nn.Module):
    def __init__(self, n_in_feat, n_head = 8, n_head_feat = 64, dropout = 0.):
        super().__init__()
        n_hidden_feat = n_head_feat *  n_head
        n_out_feat = n_in_feat
        project_out = not (n_head == 1 and n_head_feat == n_in_feat)

        self.n_head = n_head
        self.scale = n_head_feat ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(n_in_feat, n_hidden_feat * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(n_hidden_feat, n_out_feat),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(input).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_head), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)