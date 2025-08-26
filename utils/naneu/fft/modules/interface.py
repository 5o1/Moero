# from abc import abstractmethod
# import torch
# from torch import nn
# from typing import Sequence, Union
# from torch import SymInt

# class NufftInterface(nn.Module):
#     def __init__(self, image_size: Sequence[Union[int, SymInt]], traj: torch.Tensor, kmax:float = 0.5):
#         super().__init__()
#         self.image_size = image_size[-2:]
#         self.traj = traj
#         self.kmax = kmax

#     @abstractmethod
#     def prepare(self, traj: torch.Tensor)-> torch.Tensor:
#         pass

#     @abstractmethod
#     def A(self, x: torch.Tensor)-> torch.Tensor:
#         pass
    
#     @abstractmethod
#     def At(self, x: torch.Tensor) -> torch.Tensor:
#         pass

#     def forward(self):
#         return NotImplementedError("Please use forward and backward operators.")

#     def mulchan2single(self, images: torch.Tensor, sensitivity_maps: torch.Tensor):
#         conj_sensitivity = torch.conj(sensitivity_maps)
#         numerator = torch.sum(conj_sensitivity * images, axis=1, keepdim = True)
#         denominator = torch.sum(torch.abs(sensitivity_maps) ** 2, axis=1, keepdim = True)
#         denominator = torch.where(denominator == 0, 1e-8, denominator)
#         combined_image = numerator / denominator
#         return combined_image

#     def get_float_dtype(self, dtype: torch.dtype):
#         if dtype == torch.complex64 or dtype == torch.float32:
#             return torch.float32
#         elif dtype == torch.complex128 or dtype == torch.float64:
#             return torch.float64
#         else:
#             raise ValueError("Invalid type")
        
#     def get_complex_dtype(self, dtype: torch.dtype):
#         if dtype == torch.float32 or dtype == torch.complex64:
#             return torch.complex64
#         elif dtype == torch.float64 or dtype == torch.complex128:
#             return torch.complex128
#         else:
#             raise ValueError("Invalid type")
        
#     def min_max_normalize_complex(self, x:torch.Tensor):
#         x_abs = torch.abs(x)
#         x_angle = torch.angle(x)
#         x_abs_max = torch.max(x_abs)
#         x_abs_min = torch.min(x_abs)
#         scale = x_abs_max - x_abs_min
#         bias = x_abs_min
#         x_abs = (x_abs - bias) / scale
#         return x_abs * torch.exp(1j * x_angle), scale, bias
    
#     def min_max_recover_complex(self, x:torch.Tensor, scale: float, bias: float):
#         x_abs = torch.abs(x)
#         x_angle = torch.angle(x)
#         x_abs = x_abs * scale + bias
#         return x_abs * torch.exp(1j * x_angle)

#     def z_score_normalize_complex(self, x:torch.Tensor):
#         bias = torch.mean(x)
#         scale = torch.std(x)
#         x = (x - bias) / scale
#         return x, scale, bias
    
#     def z_score_recover_complex(self, x:torch.Tensor, scale: float, bias: float):
#         x = x * scale + bias
#         return x

