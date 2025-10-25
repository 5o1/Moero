import torch
from torch import nn
import math
from ..modules.attention import TransformerBlock

class AngleFourierEncoding(nn.Module):
    """Angle -> multi-frequency sin/cos -> linear -> d_model."""
    freq: torch.Tensor
    def __init__(self, d_model, num_frequencies=8):
        super().__init__()
        self.M = num_frequencies
        self.proj = nn.Linear(2 * self.M, d_model)
        # Cache frequencies as buffer to avoid re-allocating
        self.register_buffer('freq', torch.arange(1, self.M + 1, dtype=torch.float32).view(1, 1, -1), persistent=False)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: [B, N] in radians (or [B, N, 1]). Will be wrapped to (-pi, pi].
        Return: [B, N, d_model]
        """
        # Squeeze last dim if it's 1
        if theta.dim() == 3 and theta.size(-1) == 1:
            theta = theta.squeeze(-1)
        if theta.dim() != 2:
            raise ValueError(f"theta must be [B, N] (or [B, N, 1]), but got {theta.shape}")

        # Use proj dtype for trig to avoid dtype mismatch with Linear weights
        dtype_proj = self.proj.weight.dtype
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        theta_fp = theta.to(dtype_proj)  # better precision control

        # Broadcast frequencies
        if self.freq.device != theta_fp.device or self.freq.dtype != dtype_proj:
            freq = torch.arange(1, self.M + 1, device=theta_fp.device, dtype=dtype_proj).view(1, 1, -1)
        else:
            freq = self.freq

        ang = theta_fp.unsqueeze(-1) * freq  # [B, N, M]
        emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)  # [B, N, 2M]
        # Optional: scale to stabilize
        # emb = emb / math.sqrt(2 * self.M)

        emb = self.proj(emb)  # [B, N, d_model]
        # Cast to x dtype if needed
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)

        return x + emb


class SelfFiLM(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.gamma_proj = nn.Linear(n_feat, n_feat)
        self.beta_proj = nn.Linear(n_feat, n_feat)
        


class Transformer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            enc_channels: int,
            dec_channels: int,
            n_head: int,
            n_head_feat: int,
            mlp_ratio: int = 4,
            n_enc: int = 4,
            n_dec: int = 4,
            dropout: float = 0.0,
        ):
        super().__init__()
        self.pe_layer_enc = AngleFourierEncoding(d_model=enc_channels)
        self.in_to_enc = nn.Linear(in_channels, enc_channels)
        self.encoders = nn.ModuleList([
            TransformerBlock(
                n_feat=enc_channels,
                n_head=n_head,
                n_head_feat=n_head_feat,
                n_mlp_feat=enc_channels * mlp_ratio,
                dropout=dropout
            ) for _ in range(n_enc)
        ])
        self.enc_to_dec = nn.Linear(enc_channels, dec_channels)
        self.in_to_dec = nn.Linear(in_channels, dec_channels)

        self.pe_layer_dec = AngleFourierEncoding(d_model=dec_channels)
        self.decoders = nn.ModuleList([
            TransformerBlock(
                n_feat=dec_channels,
                n_head=n_head,
                n_head_feat=n_head_feat,
                n_mlp_feat=dec_channels * mlp_ratio,
                dropout=dropout
            ) for _ in range(n_dec)
        ])
        self.dec_to_out = nn.Linear(dec_channels, in_channels)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (B, n, c)
        x_dec: (B, m, c)
        pos:   (B, n + m, d_pos), where d_pos==1 if it's a scalar angle per token.
               If d_pos > 1, extract/compute an angle before passing in.
        Return: (B, m, c)
        """
        B, n, _ = x_enc.shape
        _, m, _ = x_dec.shape

        # Encoder path
        x_enc_embed = self.in_to_enc(x_enc)
        pos_enc = pos[:, :n]  # (B, n, d_pos)
        x_enc_embed = self.pe_layer_enc(x_enc_embed, pos_enc)
        for encoder in self.encoders:
            x_enc_embed = encoder(x_enc_embed)

        # Project encoder memory to decoder feature space
        mem = self.enc_to_dec(x_enc_embed)  # (B, n, dec_channels)

        # Decoder queries
        x_dec_embed = self.in_to_dec(x_dec)  # (B, m, dec_channels)
        pos_dec = pos[:, n:]  # (B, m, d_pos)
        x_dec_embed = self.pe_layer_dec(x_dec_embed, pos_dec)

        # Concatenate memory and queries, let self-attention act as "cross-attention"
        x_all = torch.cat([mem, x_dec_embed], dim=1)  # (B, n+m, dec_channels)
        for decoder in self.decoders:
            x_all = decoder(x_all)

        x_out = self.dec_to_out(x_all[:, n:, :])  # (B, m, in_channels)
        x_out = torch.cat([x_enc, x_out], dim=-2)
        return x_out


        