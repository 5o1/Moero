import torch
from torch import nn
from torch_kmeans import KMeans, CosineSimilarity, DotProductSimilarity, ClusterResult
from . import dist as dist_fn
from torch import LongTensor, Tensor
from utils.naneu.helpers.context import register_extra_metric, register_extra_loss
from einops.layers.torch import Rearrange
from math import ceil
from .dist import all_reduce
from typing import Tuple


class BranchNav(torch.nn.Module):
    route_cnt: Tensor
    pass

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
    

class LearnableBranchNav(BranchNav):
    route_ema: torch.Tensor

    def __init__(
            self,
            in_channels: int,
            top_k: int = 2,
            poolsize: int = 8,
            eps = 1e-13,
            ema_decay: float = 0.99,
            balance_lambda: float = 1.0,
            z_loss_coef: float = 1e-3,
            aux_loss_coef: float = 1e-2,
            idx: int = 0,
        ):
        super().__init__()
        assert poolsize >= 1, "pool_size must be >= 1"
        assert 1 <= top_k <= poolsize, "top_k must be in [1, pool_size]"

        self.eps = eps
        self.balance_lambda = balance_lambda
        self.ema_decay = ema_decay
        self.top_k = top_k
        self.poolsize = poolsize
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.idx = idx

        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, poolsize),
        )
        self.wnoise = nn.Linear(in_channels, poolsize)
        self.register_buffer("route_ema", torch.zeros((poolsize,), dtype=torch.float32))

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _balancing_prior(self) -> Tensor:
        """
        Build additive log-prior for logits from EMA usage:
        log_prior_i = -log(usage_i + eps), scaled by balance_lambda.
        """
        # Normalize EMA to a rate-like vector (sum to 1) when non-zero
        usage = self.route_ema.clamp_min(self.eps)
        usage = usage / usage.sum().clamp_min(self.eps)  # make it comparable across batch sizes
        log_prior = -torch.log(usage)                    # higher for rarely used experts
        return self.balance_lambda * log_prior           # [pool_size]


    def focus_metric(self, route_count: Tensor) -> Tensor:
        """
        Calculate focus metric from route counts. Entropy-based, normalized to [0, 1].
        """
        route_count = route_count.float()
        route_freq = route_count / (route_count.sum() + self.eps)
        entropy = -(route_freq * (route_freq + self.eps).log()).sum()
        entropy_max = torch.log(torch.tensor(float(self.poolsize), device=route_count.device))
        entropy_norm = entropy / (entropy_max + self.eps)
        focus = 1.0 - entropy_norm
        return focus

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[Tensor, Tensor, LongTensor]:
        x_detached = x.detach()

        # CNN -> GAP -> logits
        logits: torch.Tensor = self.head(x)        # [B, pool_size]
        logits_detached: torch.Tensor = self.head(x_detached)  # [B, pool_size]

        # z-loss
        if self.training and self.poolsize > 1 and self.z_loss_coef > 0.0:
            z_loss = torch.logsumexp(logits_detached.float(), dim=-1) ** 2 * self.z_loss_coef  # [B]
            register_extra_loss(self, f"route_zloss{self.idx}", z_loss.mean())

        # Add EMA-based balancing prior in logits space
        prior = self._balancing_prior().unsqueeze(0).to(logits.device, logits.dtype)
        logits = logits + prior
        logits_detached = logits_detached + prior

        if self.training:
            logits_topk = logits + torch.randn_like(logits) * (nn.functional.softplus(self.wnoise(x)) + self.eps)  # [B, pool_size]
        else:
            logits_topk = logits

        # Compute softmax over all experts (differentiable for every logit)
        prob_all = torch.softmax(logits.float(), dim=-1).to(logits.dtype)  # [B, pool_size]
        prob_all_detached = torch.softmax(logits_detached.float(), dim=-1).to(logits_detached.dtype)  # [B, pool_size]

        # auxiliary loss
        if self.training and self.poolsize > 1 and self.aux_loss_coef > 0.0:
            route_count = self.route_ema.float()
            route_freq = route_count / (route_count.sum() + self.eps)
            aux_loss = (route_freq * prob_all_detached).sum(dim=-1).mean() * self.aux_loss_coef
            register_extra_loss(self, f"route_auxloss{self.idx}", aux_loss)
            

        # Top-k indices/mask for hard dispatch
        topk = torch.topk(logits_topk, k=self.top_k, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(logits).scatter_(dim=-1, index=topk.indices, value=1.0)  # [B, pool_size]

        # Sparse weights = soft probabilities masked by top-k
        prob_weights = prob_all * topk_mask  # [B, pool_size]

        # Normalize sparse weights to sum to 1 over top-k
        denom = prob_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps).detach()  # stop-grad
        prob_weights = prob_weights / denom  # [B, pool_size]

        # Update frequency stats
        if self.training and self.poolsize > 1:
            with torch.no_grad():
                batch_select = topk_mask.sum(dim=0) # [pool_size], how many times each expert chosen
                all_reduce(batch_select)
                self.route_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_select) # EMA update

                # Logging
                focus_score = self.focus_metric(self.route_ema)
                register_extra_metric(self, f"route_focus{self.idx}", focus_score, op="mean")

        return prob_weights, topk_mask, topk.indices