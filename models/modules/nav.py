import torch
from torch_kmeans import KMeans, CosineSimilarity, DotProductSimilarity, ClusterResult
from . import dist as dist_fn
from torch import LongTensor, Tensor
from utils.naneu.helpers.context import register_extra_metric


class BranchNav(torch.nn.Module):
    pass

# Bad attempt
# class GMMBranchNav(torch.nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             n_components: int = 2,
#             buffer_maxsize: int = 128,
#             cluster_niter: int = 100,
#             cluster_niter_after_init: int = 100,
#             fixed_cluster: bool = True,
#             save_dir= str,
#     ):
#         super().__init__()
#         self.n_components = n_components
#         self.cluster = GaussianMixture(n_components=n_components, n_features=in_channels)
#         self.norm = torch.nn.LayerNorm(in_channels, eps = 1e-13, elementwise_affine=True)
#         self.cluster_niter = cluster_niter
#         self.cluster_niter_after_init = cluster_niter_after_init
#         self.cluster_init_args = self.cluster.init_args_clone()
#         self.fixed_cluster = fixed_cluster
#         self.save_dir = save_dir

#         self.buffer = []
#         self.buffer_cnt = 0
#         self.buffer_maxsize = buffer_maxsize

#         self.route_cnt = torch.zeros((self.n_components,))

#     def balance_route(self) -> torch.Tensor:
#         route_max, idx_max  = self.route_cnt.max(0)
#         route_min, idx_min  = self.route_cnt.min(0)

#         if route_max / (route_min + 1e-13) > 3:
#             return idx_min
#         return None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.norm(x)

#         if self.training and not (self.cluster.is_fitted() and self.fixed_cluster):
#             self.buffer.append(x)
#             self.buffer_cnt += x.size(0)

#             buffer_local_size = self.buffer_maxsize // dist_fn.get_world_size()

#             if self.buffer_cnt > buffer_local_size:
#                 local_buffer = torch.cat(self.buffer, dim=0)
#                 global_buffer = dist_fn.all_gather(local_buffer)
#                 global_buffer = [buffer.to(x) for buffer in global_buffer]
#                 global_buffer = torch.cat(global_buffer, dim=0)

#                 niter = self.cluster_niter_after_init if self.cluster.is_fitted() else self.cluster_niter

#                 if not self.fixed_cluster:
#                     self.cluster.init_args_restore(self.cluster_init_args)

#                 self.cluster.fit(global_buffer, n_iter=niter)

#                 if self.cluster.is_fitted() and not self.fixed_cluster:
#                     self.cluster_init_args = self.cluster.init_args_clone()

#                 self.buffer.clear()
#                 self.buffer_cnt = 0
            
#         route = self.balance_route()
#         if route is None and self.cluster.is_fitted():
#             unique_values, counts = torch.unique(self.cluster(x), return_counts=True)
#             max_count = counts.max()
#             max_value = unique_values[counts.argmax()]
#             route = max_value
#         else:
#             _, route = self.route_cnt.min(0)
        
#         self.route_cnt[route] += 1
        
#         route_mask = torch.zeros(self.n_components, dtype = torch.bool, device=x.device)
#         route_mask[route] = True
#         return route_mask

class KmeansBranchNav(BranchNav):
    sample_idx: torch.Tensor
    
    labels: LongTensor
    centers: Tensor
    inertia: Tensor
    x_org: Tensor
    x_norm: Tensor
    k: LongTensor
    soft_assignment: Tensor
    is_fitted: torch.Tensor

    def __init__(
            self,
            in_channels: int,
            n_components: int = 2,
            buffer_maxsize: int = 128,
            buffer_nskip:int = 2000,
            cluster_niter: int = 100,
            fixed_cluster: bool = True,
            seed: int = 42,
            ckpt_bias = 0 # Equals to `(buffer_maxsize // n_ddp_ranks) * n_ddp_ranks - buffer_maxsize`
    ):
        super().__init__()
        self.n_components = n_components
        self.cluster = KMeans(n_clusters=n_components, distance=CosineSimilarity, verbose=False, seed = seed)

        self.cluster_niter = cluster_niter
        self.fixed_cluster = fixed_cluster

        self.buffer = []
        self.buffer_cnt = 0
        self.buffer_maxsize = buffer_maxsize
        self.buffer_nskip = buffer_nskip
        self.register_buffer("sample_idx", torch.tensor(0, dtype=torch.int64))

        self.route_cnt = torch.zeros((self.n_components,))

        # Buffers for cluster results
        self.register_buffer("labels", torch.empty(1, buffer_maxsize + ckpt_bias))
        self.register_buffer("centers", torch.empty(1, self.n_components, in_channels))
        self.register_buffer("inertia", torch.empty(1))
        self.register_buffer("x_org", torch.empty(1, buffer_maxsize + ckpt_bias, in_channels))
        self.register_buffer("x_norm", torch.empty(1, buffer_maxsize + ckpt_bias, in_channels))
        self.register_buffer("k", torch.empty(1))
        self.register_buffer("soft_assignment", torch.empty(0))
        self.register_buffer("is_fitted", torch.tensor(0, dtype=bool))

    def balance_route(self) -> torch.Tensor:
        route_max, idx_max  = self.route_cnt.max(0)
        route_min, idx_min  = self.route_cnt.min(0)

        if route_max / (route_min + 1e-13) > 3:
            return idx_min
        return None
    

    def sync_cluster(self):
        if self.is_fitted != self.cluster.is_fitted and self.is_fitted:
            cluster_result = ClusterResult(
                        labels=self.labels,
                        centers = self.centers,
                        inertia=self.inertia,
                        x_org = self.x_org,
                        x_norm = self.x_norm,
                        k = self.k,
                        soft_assignment=None if self.soft_assignment.numel() == 0 else self.soft_assignment
                    )
            self.cluster._result=cluster_result
            print("BranchNav Cluster loaded with centers:", self.centers.shape, "and labels:", self.labels.shape, "inertia:", self.inertia)

    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     res = super()._load_from_state_dict(state_dict, *args, **kwargs)
    #     self.sync_cluster()
    #     return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.sync_cluster()
        self.sample_idx += x.size(0)

        if self.training and not (self.cluster.is_fitted and self.fixed_cluster) and self.sample_idx > self.buffer_nskip:
            self.buffer.append(x.detach())
            self.buffer_cnt += x.size(0)

            buffer_local_size = self.buffer_maxsize // dist_fn.get_world_size()

            if self.buffer_cnt >= buffer_local_size:
                local_buffer = torch.cat(self.buffer, dim=0)
                global_buffer = dist_fn.all_gather(local_buffer)
                global_buffer = torch.cat(global_buffer, dim=0)

                global_buffer = global_buffer.unsqueeze(0)

                self.cluster.fit(global_buffer)

                if self.cluster.is_fitted: # Sync kernel state
                    result: ClusterResult = self.cluster._result
                    dist_fn.check_sync(result.centers, "BranchNav Cluster Centers")

                    self.labels=result.labels
                    self.centers=result.centers
                    self.inertia=result.inertia
                    self.x_org=result.x_org
                    self.x_norm=result.x_norm
                    self.k=result.k
                    self.soft_assignment=torch.empty(0, device = self.soft_assignment.device) if result.soft_assignment is None else result.soft_assignment
                    self.is_fitted.fill_(1)

                    register_extra_metric(self, "branchnav_kmeans_centers", self.inertia, op = "mean")

                self.buffer.clear()
                self.buffer_cnt = 0
            
        route = self.balance_route()
        if route is None and self.cluster.is_fitted:
            center_labels = self.cluster.predict(x.unsqueeze(0)).squeeze(0)
            if (center_labels > self.n_components - 1).any():
                raise ValueError("Cluster labels exceed the number of components.")
            unique_values, counts = torch.unique(center_labels, return_counts=True)
            max_count = counts.max()
            max_value = unique_values[counts.argmax()]
            route = max_value
        else:
            _, route = self.route_cnt.min(0)

        self.route_cnt[route] += 1

        route_mask = torch.zeros(self.n_components, dtype = torch.bool, device=x.device)
        route_mask[route] = 1
        return route_mask