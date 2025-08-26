# import sys
# import os
# sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
# from bart import bart

# import torch
# import numpy as np
# from typing import Sequence, Union
# from torch import SymInt

# from .interface import NufftInterface

# class BARTSampling2d(NufftInterface):
#     def __init__(self, image_size: Sequence[Union[int, SymInt]], traj: torch.Tensor):
#         super().__init__(image_size, traj)

#     def prepare(self, traj: torch.Tensor)-> torch.Tensor:
#         pass

#     def A(self, x: torch.Tensor)-> torch.Tensor:
#         pass
    
#     def At(self, x: torch.Tensor) -> torch.Tensor:
#         pass

#     def tensor2bart(self, x: torch.Tensor) -> np.ndarray:
#         pass

#     def bart2tensor(self, x: np.ndarray) -> torch.Tensor:
#         pass

#     def tensor2bart_noncart(self, x : torch.Tensor) -> np.ndarray:
#         pass

#     def bart2tensor_noncart(self, x: np.ndarray) -> torch.Tensor:
#         pass
