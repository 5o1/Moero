"""
Modified from https://github.com/ternencewu123/PARCEL/blob/main/PARCEL/net/net_parts.py
"""


import torch
from torch import nn
from typing import Callable

class ConjugateGradient(nn.Module):
    def __init__(self, tol: float = 1e-8, max_iter: int = 10, ndim = 2):
        super().__init__()
        self.tol = tol
        self.max_iter = max_iter
        self.ndim = ndim

        assert ndim > 0, "ndim must be greater than 0"

    def forward(self, b: torch.Tensor, AtA: Callable[[torch.Tensor], torch.Tensor], lam: float) -> torch.Tensor:
        x = torch.zeros_like(b)
        r, p = b, b
        rTr = torch.sum(torch.conj(r) * r, dim=list(range(-self.ndim, 0, 1))).real
        for _ in range(self.max_iter):
            Ap = AtA(p) + lam * p
            alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=list(range(-self.ndim, 0, 1))).real
            x = x + alpha.view(alpha.size(0), *([1]*(x.ndim-1))) * p
            r = r - alpha.view(alpha.size(0), *([1]*(r.ndim-1))) * Ap
            rTrNew = torch.sum(torch.conj(r) * r, dim=list(range(-self.ndim, 0, 1))).real
            if rTrNew.max() < self.tol:
                break
            beta = rTrNew / rTr
            rTr = rTrNew
            p = r + beta.view(beta.size(0), *([1]*(r.ndim-1))) * p
        return x