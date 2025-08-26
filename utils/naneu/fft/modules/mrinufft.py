# from mrinufft import get_density, get_operator, initialize_2D_spiral
# from mrinufft.operators import FourierOperatorBase
# import numpy as np
# import torch


# from .interface import NufftInterface

# class MRINUFFTSampling2d(NufftInterface):
#     nufft_ob: FourierOperatorBase
#     traj: torch.Tensor

#     def __init__(self, image_size, traj_kwargs, n_batchs: int = 10, n_coils: int = 1):
#         super().__init__(image_size)
#         self.traj_kwargs = traj_kwargs
#         self.n_batchs = n_batchs
#         self.n_coils = n_coils
#         self.traj = self.get_traj(traj_kwargs)
#         self.density = get_density("voronoi", self.traj, shape=image_size, osf=2.0)
#         self.nufft_ob = get_operator("cufinufft")(
#             self.traj,
#             image_size,
#             density = self.density,
#             n_batchs = n_batchs,
#             n_coils = n_coils,
#             squeeze_dims=False,
#             upsampfac = 2.0
#         )

#     def get_traj(self, traj_kwargs: torch.Tensor):
#         return initialize_2D_spiral(**traj_kwargs).astype(np.float32)

#     def nufft(self, x: torch.Tensor):
#         # b c h w
#         b, c, h, w = x.shape
#         p, r, _ = self.traj.shape
#         self.nufft_ob.n_batchs = b
#         self.nufft_ob.n_coils = c
#         x = self.nufft_ob.op(x)
#         x = x.view(b,c, p, r)
#         return x
    
#     def nuifft(self, x):
#         x = x.view(x.shape[0], x.shape[1], -1)
#         self.nufft_ob.n_batchs = x.shape[0]
#         self.nufft_ob.n_coils = x.shape[1]
#         x = self.nufft_ob.adj_op(x) # b kdata
#         return x