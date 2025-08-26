# """
# Modified from https://github.com/ternencewu123/PARCEL/blob/main/PARCEL/net/net.py
# """

# from torch import nn
# import myf
# import torch
# from typing import Callable

# class MoDL(nn.Module):
#     def __init__(self, model:nn.Module, cg: nn.Module, lam = 0.05, n_unrolled = 5):
#         super(MoDL, self).__init__()
#         self.layers = model
#         self.lam = nn.Parameter(torch.FloatTensor([lam]))
#         self.cg = cg
#         self.n_unrolled = n_unrolled

#     def forward(self, input: torch.Tensor, AtA: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
#         x = input
#         for _ in range(self.n_unrolled):
#             x = self.layers(x)
#             x = input + self.lam * x
#             x = myf.real_to_complex(x)
#             x = self.cg(x, AtA, self.lam)
#             x = myf.complex_to_real(x)
#         return x