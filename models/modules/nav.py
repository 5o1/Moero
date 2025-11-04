import torch
from torch import nn
from torch_kmeans import KMeans, CosineSimilarity, DotProductSimilarity, ClusterResult, LpDistance
from . import dist as dist_fn
from .conv import CAB, CABChain
from torch import LongTensor, Tensor
from utils.naneu.helpers.context import register_extra_metric, register_extra_loss, register_extra_output
from einops.layers.torch import Rearrange
from math import ceil
from typing import Tuple, List, Literal
from functools import partial
from torch.nn import functional as F
import math
from data.transforms.crop import center_crop_to_smallest
from enum import Enum, auto


class BranchNav(torch.nn.Module):
    route_cnt: Tensor
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
            eps = 1e-5,
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
            nn.PReLU(),
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


    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[Tensor, Tensor, LongTensor]:
        # CNN -> GAP -> logits
        logits: torch.Tensor = self.head(x)        # [B, pool_size]

        # z-loss
        if self.training and self.poolsize > 1 and self.z_loss_coef > 0.0:
            z_loss = torch.logsumexp(logits.float(), dim=-1) ** 2 * self.z_loss_coef  # [B]
            register_extra_loss(self, f"route_zloss{self.idx}", z_loss.mean())

        # Add EMA-based balancing prior in logits space
        prior = self._balancing_prior().unsqueeze(0).to(logits.device, logits.dtype)
        logits = logits + prior

        if self.training:
            logits_topk = logits + torch.randn_like(logits) * (nn.functional.softplus(self.wnoise(x)) + self.eps)  # [B, pool_size]
        else:
            logits_topk = logits

        # Compute softmax over all experts (differentiable for every logit)
        prob_all = torch.softmax(logits.float(), dim=-1).to(logits.dtype)  # [B, pool_size]

        # auxiliary loss
        if self.training and self.poolsize > 1 and self.aux_loss_coef > 0.0:
            route_count = self.route_ema.float()
            route_freq = route_count / (route_count.sum() + self.eps)
            aux_loss = (route_freq * prob_all).sum(dim=-1).mean() * self.aux_loss_coef
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
                dist_fn.all_reduce(batch_select)
                self.route_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_select) # EMA update

                # Logging
                focus_score = self.focus_metric(self.route_ema)
                register_extra_metric(self, f"route_focus{self.idx}", focus_score, op="mean")

        return prob_weights, topk_mask, topk.indices


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            pyramid_channels : List[int], # depth + 1
            embedding_channels: int,
            ):
        super().__init__()
        self.level0 = nn.Sequential(
                nn.Conv2d(in_channels, pyramid_channels[0], kernel_size=1, padding=0, bias=False),
                nn.InstanceNorm2d(pyramid_channels[0], affine=True, eps=1e-5),
                nn.PReLU(),
                CAB(pyramid_channels[0], kernel_size=3, reduction=4, dropout=0.0)
            )
        self.depth = len(pyramid_channels) - 1
        self.levels = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(len(pyramid_channels) - 1):
            ci, co = pyramid_channels[i], pyramid_channels[i + 1]
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(ci, co, kernel_size=3, stride=2, padding=1, bias=True),
                    nn.PReLU(),
                )
            )

            self.levels.append(CABChain(co, n_cab=2, kernel_size=3, reduction=4, dropout=0.0, is_res=True))

        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(pyramid_channels[-1], embedding_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.level0(x)

        for i in range(self.depth):
            x = self.down[i](x)
            x = self.levels[i](x)

        x = self.embedding(x)
        return x

class FeatExtractBranchNav(BranchNav):
    route_ema: torch.Tensor
    weights_ema: torch.Tensor
    global_step: torch.Tensor
    current_temperature: torch.Tensor
    current_aux_loss_coef: torch.Tensor
    current_aux_loss_b_coef: torch.Tensor
    route_ema_val: torch.Tensor
    weights_ema_val: torch.Tensor
    prototype: torch.Tensor

    def __init__(
            self,
            # ------------------------------------------------
            # Classifier parameters
            in_channels: int,
            pyramid_channels : List[int], # depth + 1
            embedding_channels: int,
            cluster: Literal["linear", "prototype"] = "prototype",
            prototype_act : Literal["softmax", "energy"] = "softmax",
            use_residual_expert: bool = False,
            # ------------------------------------------------
            # Routing parameters
            top_k: int = 2, # number of experts to select
            poolsize: int = 8, # number of experts
            ema_decay: float = 0.99, # for counters EMA
            #------------------------------------------------
            # balancing parameters
            balance_prior_lambda: float = 0.0,
            z_loss_coef: float = 1e-3,
            aux_loss_coef: float = 1e-1,
            aux_loss_batch_coef: float = 1e-1,
            prob_sparse_loss_coef: float = 1e-3,
            # ------------------------------------------------
            # Auxiliary losses annealing parameters
            aux_loss_coef_min: float = 1e-3,      #    minimum aux loss coef
            aux_loss_batch_coef_min: float = 1e-3, #    minimum aux loss batch coef
            aux_loss_decay_type: Literal["linear", "cosine", "exp"] = "linear",
            aux_loss_step0: int = 20000,        #    starting step for annealing
            aux_loss_stepn: int = 40000,        #    ending step for annealing
            aux_loss_gamma: float = 0.9995,        #    for exp: coef_t = max(coef_min, coef_init * gamma^t)
            # ------------------------------------------------
            # Temperature annealing parameters
            temp_init: float = 1.0,        #    initial temperature
            temp_min: float = 0.0,         #    minimum temperature
            temp_decay_type: Literal["linear", "cosine", "exp"] = "linear",
            temp_decay_steps: int = 10000, #    number of steps for decay
            temp_gamma: float = 0.9995,    #    for exp: tau_t = max(temp_min, temp_init * gamma^t)
            # ------------------------------------------------
            eps = 1e-5,
            idx: int = 0, # identifier for multiple nav modules
            detach: bool = True, # whether to detach input features
            n_skip: int = 0, # number of initial steps, all of the classifiers will not be used before n_skip
            batch_cache_size: int = 20, # number of previous batches to cache for dummy batch creation
            # ------------------------------------------------
            freezed: bool = False, # whether to freeze the parameters
        ):
        super().__init__()
        assert poolsize >= 1, "pool_size must be >= 1"
        assert 1 <= top_k <= poolsize, "top_k must be in [1, pool_size]"
        assert not (cluster == "prototype" and balance_prior_lambda != 0.0), "Cannot use balancing prior with prototype clustering."
        assert z_loss_coef >= 0.0, "z_loss_coef must be non-negative."
        assert aux_loss_coef >= 0.0, "aux_loss_coef must be non-negative."
        assert aux_loss_batch_coef >= 0.0, "aux_loss_batch_coef must be non-negative."
        assert prob_sparse_loss_coef >= 0.0, "prob_sparse_loss_coef must be non-negative."
        if use_residual_expert:
            assert poolsize > 1 and top_k > 1, "Residual expert requires poolsize > 1 and top_k > 1."

        self.depth = len(pyramid_channels) - 1

        self.eps = eps
        self.balance_prior_lambda = balance_prior_lambda
        self.ema_decay = ema_decay
        self.top_k = top_k
        self.poolsize = poolsize
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_batch_coef = aux_loss_batch_coef
        self.prob_sparse_loss_coef = prob_sparse_loss_coef
        self.n_skip = n_skip
        self.idx = idx
        self.detach = detach
        self.batch_cache_size = batch_cache_size
        self.cluster = cluster
        self.prototype_act = prototype_act
        self.use_residual_expert = use_residual_expert

        if self.poolsize > 1:
            if self.use_residual_expert:
                self.poolsize_logits = poolsize - 1
                self.top_k_logits = top_k - 1
            else:
                self.poolsize_logits = poolsize
                self.top_k_logits = top_k
            # ---------------- Classifier ----------------
            self.extractors = Encoder(in_channels, pyramid_channels, embedding_channels)
            self.head = nn.Sequential(
                nn.Linear(embedding_channels, 256, bias=False),
                nn.LayerNorm(256),
                nn.PReLU(),
                nn.Linear(256, 64, bias=False),
                nn.LayerNorm(64),
                nn.PReLU(),
                nn.Linear(64, self.poolsize_logits),
            )
            if self.cluster == "prototype":
                proto = torch.empty((self.poolsize_logits, self.poolsize_logits))
                proto = torch.nn.init.orthogonal_(proto)
                self.register_buffer("prototype", F.normalize(proto, p=2, dim=0)) # Note: prototype has been transposed for matmul

            self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
            # ---------------- Frequency Counters ----------------
            self.register_buffer("weights_ema", torch.zeros((self.poolsize_logits,), dtype=torch.float32))
            self.register_buffer("route_ema", torch.zeros((self.poolsize_logits,), dtype=torch.float32))
            self.register_buffer("weights_ema_val", torch.zeros((self.poolsize_logits,), dtype=torch.float32))
            self.register_buffer("route_ema_val", torch.zeros((self.poolsize_logits,), dtype=torch.float32))
            # ---------------- Noise Temperature Schedule ----------------
            self.temp_init = float(temp_init)
            self.temp_min = float(temp_min)
            self.temp_decay_type = temp_decay_type
            self.temp_decay_steps = int(temp_decay_steps)
            self.temp_gamma = float(temp_gamma)
            self.register_buffer("current_temperature", torch.tensor(self.temp_init, dtype=torch.float32))
            # ---------------- Auxiliary Losses Schedule ----------------
            self.aux_loss_coef_min = float(aux_loss_coef_min)
            self.aux_loss_batch_coef_min = float(aux_loss_batch_coef_min)
            self.aux_loss_decay_type = aux_loss_decay_type
            self.aux_loss_step0 = int(aux_loss_step0)
            self.aux_loss_stepn = int(aux_loss_stepn)
            self.aux_loss_gamma = float(aux_loss_gamma)
            self.register_buffer("current_aux_loss_coef", torch.tensor(self.aux_loss_coef, dtype=torch.float32))
            self.register_buffer("current_aux_loss_b_coef", torch.tensor(self.aux_loss_batch_coef, dtype=torch.float32))
            # ---------------- Batch cache for dummy batch ----------------
            self.batch_cache = []

        self.apply(self._init_weights)

        if self.freezed:
            for param in self.parameters():
                param.requires_grad = False

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    # ---------------- temperature scheduler ----------------
    @torch.no_grad()
    def cal_temperature(self, step: int | None = None) -> torch.Tensor:
        """
        Calculate temperature tau at given step according to the annealing schedule.
        Args:
            step (int, optional): Current training step. If None, use self.global_step.
        Returns:
            torch.Tensor: Scalar tensor representing the temperature tau.
        """
        # Returns a scalar tensor tau on the same device as route_ema
        device = self.route_ema.device
        if step is None:
            step = int(self.global_step.item())
        s = float(step)

        if step < self.n_skip:
            tau = self.temp_init
            return torch.tensor(tau, device=device, dtype=torch.float32)
        s = s - self.n_skip  # start decay after n_skip steps

        if self.temp_decay_type == "linear":
            # tau = temp_min + (temp_init - temp_min) * max(0, 1 - s/steps)
            if self.temp_decay_steps <= 0:
                frac = 0.0
            else:
                frac = max(0.0, 1.0 - s / float(self.temp_decay_steps))
            tau = self.temp_min + (self.temp_init - self.temp_min) * frac
        elif self.temp_decay_type == "cosine":
            # tau = temp_min + 0.5*(temp_init - temp_min)*(1 + cos(pi * min(1, s/steps)))
            if self.temp_decay_steps <= 0:
                cosw = -1.0
            else:
                x = min(1.0, s / float(self.temp_decay_steps))
                cosw = math.cos(math.pi * x)
            tau = self.temp_min + 0.5 * (self.temp_init - self.temp_min) * (1.0 + cosw)
        else:  # "exp"
            # tau = max(temp_min, temp_init * gamma^s)
            tau = max(self.temp_min, self.temp_init * (self.temp_gamma ** s))
        return torch.tensor(tau, device=device, dtype=torch.float32)

    @torch.no_grad()
    def cal_aux_loss_coef(self, step: int | None = None) -> torch.Tensor:
        """
        Calculate auxiliary loss coefficient at given step according to the annealing schedule.
        Args:
            step (int, optional): Current training step. If None, use self.global_step.
        Returns:
            torch.Tensor: Scalar tensor representing the auxiliary loss coefficient.
            torch.Tensor: Scalar tensor representing the auxiliary loss batch coefficient.
        """
        device = self.route_ema.device
        if step is None:
            step = int(self.global_step.item())
        s = float(step)

        # Main aux loss coef
        if step < self.aux_loss_step0:
            aux_loss_coef = self.aux_loss_coef
            aux_loss_batch_coef = self.aux_loss_batch_coef
        elif step >= self.aux_loss_stepn:
            aux_loss_coef = self.aux_loss_coef_min
            aux_loss_batch_coef = self.aux_loss_batch_coef_min
        else:
            s = float(step - self.aux_loss_step0)
            total = float(self.aux_loss_stepn - self.aux_loss_step0)
            if self.aux_loss_decay_type == "linear":
                frac = 1.0 - s / total
                aux_loss_coef = self.aux_loss_coef_min + (self.aux_loss_coef - self.aux_loss_coef_min) * frac
                aux_loss_batch_coef = self.aux_loss_batch_coef_min + (self.aux_loss_batch_coef - self.aux_loss_batch_coef_min) * frac
            elif self.aux_loss_decay_type == "cosine":
                cosw = math.cos(math.pi * (s / total))
                aux_loss_coef = self.aux_loss_coef_min + 0.5 * (self.aux_loss_coef - self.aux_loss_coef_min) * (1.0 + cosw)
                aux_loss_batch_coef = self.aux_loss_batch_coef_min + 0.5 * (self.aux_loss_batch_coef - self.aux_loss_batch_coef_min) * (1.0 + cosw)
            else:  # "exp"
                t = max(0.0, s)  # elapsed steps since step0
                aux_loss_coef = max(self.aux_loss_coef_min, self.aux_loss_coef * (self.aux_loss_gamma ** t))
                aux_loss_batch_coef = max(self.aux_loss_batch_coef_min, self.aux_loss_batch_coef * (self.aux_loss_gamma ** t))
        return torch.tensor(aux_loss_coef, device=device, dtype=torch.float32), torch.tensor(aux_loss_batch_coef, device=device, dtype=torch.float32)

    def cal_balancing_prior(self) -> Tensor:
        """
        Build additive log-prior for logits from EMA usage:
        log_prior_i = -log(usage_i + eps), scaled by balance_lambda.
        """
        # Normalize EMA to a rate-like vector (sum to 1) when non-zero
        usage = self.route_ema.clamp_min(self.eps)
        usage = usage / usage.sum().clamp_min(self.eps)  # make it comparable across batch sizes
        log_prior = -torch.log(usage)                    # higher for rarely used experts
        return self.balance_prior_lambda * log_prior           # [pool_size]

    def cal_shannon_redundancy(self, route_count: Tensor) -> Tensor:
        """
        Calculate Shannon redundancy from route counts.
        """
        route_count = route_count.float()
        route_freq = route_count / (route_count.sum() + self.eps)
        entropy = -(route_freq * (route_freq + self.eps).log()).sum()
        entropy_max = torch.log(torch.tensor(float(route_count.size(0)), device=route_count.device))
        entropy_norm = entropy / (entropy_max + self.eps)
        focus = 1.0 - entropy_norm
        return focus

    def cal_variation_coefficient(self, route_count: Tensor) -> Tensor:
        """
        Calculate coefficient of variation (std/mean) from route counts.
        """
        route_count = route_count.float()
        mean = route_count.mean()
        if mean == 0:
            return torch.tensor(0.0, device=route_count.device)
        std = route_count.std()
        return std / mean

    def forward(
        self,
        *inputs: List[torch.Tensor],
        step: int | None = None,
    ) -> Tuple[Tensor, Tensor, LongTensor]:

        if self.poolsize == 1: # Return if no experts were set up
            batch_size = inputs[0].size(0)
            prob_weights = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)
            topk_mask = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)
            indices = torch.zeros((batch_size, 1), device=x.device, dtype=torch.long)
            return prob_weights, topk_mask, indices

        if self.detach: # Detach input features to stop gradient flow if specified
            inputs = [input.detach() for input in inputs]

        inputs = center_crop_to_smallest(*inputs)
        x = torch.cat(inputs, dim=-3)  # concatenate along channel dimension

        # Create dummy batch by combining cached batches during training
        raw_batch_size = x.size(0)
        if self.training:
            dummy_batch = torch.cat(center_crop_to_smallest(*tuple(self.batch_cache + [x])), dim=0) if len(self.batch_cache) > 0 else x
            # pop oldest batch if over cache size
            if self.batch_cache_size > 0:
                self.batch_cache.append(x)
                while len(self.batch_cache) > self.batch_cache_size:
                    self.batch_cache.pop(0)
            x = dummy_batch

        # ---------------- compute logits ----------------
        if self.global_step.item() < self.n_skip:
            logits = torch.ones((x.size(0), self.poolsize_logits), device=x.device, dtype=x.dtype) / self.poolsize_logits
        else:
            x = self.extractors(x)  # [B, C]
            x: torch.Tensor = self.head(x)        # [B, embedding_channels]

            # z-loss
            if self.training and self.z_loss_coef > 0.0:
                z_loss = torch.logsumexp(x.float(), dim=-1) ** 2 * self.z_loss_coef
                register_extra_loss(self, f"route_zloss{self.idx}", z_loss.mean())

            if self.cluster == "linear":
                logits = x
            else:
                # cosine similarity with prototypes as logits
                x_norm = F.normalize(x, p=2, dim=-1)  # [B, embedding_channels]
                logits = torch.matmul(x_norm, self.prototype)  # [B, pool_size]

         # ---------------- update schedules ----------------
        if self.training:
            tau = self.cal_temperature(step)  # scalar tensor
            aux_loss_coef, aux_loss_b_coef = self.cal_aux_loss_coef(step)
            # keep a copy for logging/inspection
            with torch.no_grad():
                self.current_temperature.copy_(tau)
                self.current_aux_loss_coef.copy_(aux_loss_coef)
                self.current_aux_loss_b_coef.copy_(aux_loss_b_coef)
        else:
            # In eval, you may choose tau=1.0 to use the learned sharp distribution
            tau = torch.tensor(1.0, device=logits.device, dtype=torch.float32)

        # Add EMA-based balancing prior in logits space
        if self.balance_prior_lambda != 0.0 and self.cluster != "prototype":
            prior = self.cal_balancing_prior().unsqueeze(0).to(logits.device, logits.dtype)
            logits = logits + prior

        # ---------------- temperature-aware noise & top-k scores ----------------
        if self.training:
            # Only perform noise injection during training and during caculation of top-k scores
            logits_topk = logits + torch.randn_like(logits) * tau
        else:
            logits_topk = logits

        # Compute softmax over all experts for weights
        if self.cluster == "prototype" and self.prototype_act == "energy":
            prob_all = logits ** 2
        else:
            prob_all = torch.softmax((logits).float(), dim=-1).to(logits.dtype)  # [B, pool_size]

        # auxiliary loss for global balance
        if self.aux_loss_coef > 0.0 and self.training and self.global_step >= self.n_skip:
            route_count = self.route_ema.float()
            route_freq = route_count / (route_count.sum() + self.eps)
            aux_loss = (route_freq * prob_all).sum(dim=-1).mean() * aux_loss_coef
            register_extra_loss(self, f"global_auxloss", aux_loss, op="sum")

        # probability sparsity loss for sparsing the weights
        if self.prob_sparse_loss_coef > 0.0 and self.training and self.global_step >= self.n_skip:
            if self.cluster == "prototype" and self.prototype_act == "energy":
                scores = (logits ** 2).float()
                p = scores / (scores.sum(dim=-1, keepdim=True) + self.eps)
                logp = (p + self.eps).log()
                prob_sparse_loss = - (p * logp).sum(dim=-1).mean() * self.prob_sparse_loss_coef
            else:
                log_prob = torch.log_softmax(logits.float(), dim=-1)
                p = log_prob.exp()
                prob_sparse_loss = - (p * log_prob).sum(dim=-1).mean() * self.prob_sparse_loss_coef
            register_extra_loss(self, f"prob_sparse_loss", prob_sparse_loss, op="sum")

        # Top-k indices/mask for hard dispatch
        topk_scores = logits_topk  # reduce logit contrast when tau is large
        topk = torch.topk(topk_scores, k=self.top_k_logits, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(logits).scatter_(dim=-1, index=topk.indices, value=1.0)  # [B, pool_size]

        # auxiliary loss for batch balance
        if self.aux_loss_batch_coef > 0.0 and self.training and self.global_step >= self.n_skip:
            with torch.no_grad():
                batch_select = topk_mask.sum(dim=0) # [pool_size], how many times each expert chosen
                dist_fn.all_reduce(batch_select)
                batch_size_eff = float(x.size(0))
                batch_freq = batch_select.float() / (batch_size_eff + self.eps)
            aux_loss_batch = (batch_freq * prob_all).sum(dim=-1).mean() * aux_loss_b_coef
            register_extra_loss(self, f"batch_auxloss", aux_loss_batch, op="sum")

        # Sparse weights = soft probabilities masked by top-k
        prob_weights = prob_all * topk_mask  # [B, pool_size]

        if self.use_residual_expert:
            # Residual energy for the unselected experts
            sum_selected = prob_weights.sum(dim=-1, keepdim=True)  # [B, 1]
            prob_weights = torch.cat([prob_weights, 1.0 - sum_selected], dim=-1)  # [B, pool_size_logits + 1]
            topk_mask = torch.cat([topk_mask, torch.ones((topk_mask.size(0), 1), device=topk_mask.device, dtype=topk_mask.dtype)], dim=-1)

        # Normalize sparse weights to sum to 1 over top-k
        denom = prob_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps).detach()  # stop-grad
        prob_weights_norm = prob_weights / denom  # [B, pool_size]

        # Update stats
        with torch.no_grad():
            batch_select = topk_mask.sum(dim=0) # [pool_size], how many times each expert chosen
            weights_sum = prob_weights_norm.sum(dim=0) # [pool_size], total weights assigned to each expert
            dist_fn.all_reduce(batch_select)
            dist_fn.all_reduce(weights_sum)
            if self.training:
                self.route_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_select[:self.poolsize_logits]) # EMA update
                self.weights_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * weights_sum[:self.poolsize_logits]) # EMA update
                # Logging
                register_extra_metric(self, f"shannon_w{self.idx}", self.cal_shannon_redundancy(self.weights_ema), op="mean")
                register_extra_metric(self, f"cv_c{self.idx}", self.cal_variation_coefficient(self.route_ema), op="mean")
                register_extra_metric(self, f"temperature", self.current_temperature, op="mean")
                register_extra_metric(self, f"aux_loss_coef", self.current_aux_loss_coef, op="mean")
                register_extra_metric(self, f"aux_loss_b_coef", self.current_aux_loss_b_coef, op="mean")
                # Increment global step
                if step is None:
                    self.global_step.add_(1)
            else:
                self.route_ema_val.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_select[:self.poolsize_logits]) # EMA update
                self.weights_ema_val.mul_(self.ema_decay).add_((1 - self.ema_decay) * weights_sum[:self.poolsize_logits]) # EMA update
                # Logging
                register_extra_metric(self, f"shannon_w_val{self.idx}", self.cal_shannon_redundancy(self.weights_ema_val), op="mean")
                register_extra_metric(self, f"cv_c_val{self.idx}", self.cal_variation_coefficient(self.route_ema_val), op="mean")

        # Unpack real batch from dummy batch
        prob_weights_norm = prob_weights_norm[-raw_batch_size:, :]
        topk_mask = topk_mask[-raw_batch_size:, :]
        indices = topk.indices[-raw_batch_size:, :]
        if self.use_residual_expert:
            # add last index for residual expert
            residual_idx = torch.full((raw_batch_size, 1), self.poolsize - 1, device=indices.device, dtype=indices.dtype)
            indices = torch.cat([indices, residual_idx], dim=-1)

        return prob_weights_norm, topk_mask, indices
    
    def train(self, mode = True):
        res = super().train(mode)
        if mode: # reset val counters
            self.route_ema_val.zero_()
            self.weights_ema_val.zero_()
        return res
    

class KmeansNav2d(BranchNav):
    skip_cnt: torch.Tensor
    buffer_cnt: torch.Tensor
    route_ema: torch.Tensor
    centers: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor
    is_fitted: torch.Tensor

    def __init__(
            self,
            in_channels: int,
            top_k: int = 1,
            poolsize: int = 8,
            buffer_maxsize: int = 128,
            buffer_nskip:int = 2000,
            balance_lambda: float = 1.0,
            cluster_niter: int = 100,
            seed: int = 42,
            ema_decay: float = 0.99,
            eps = 1e-5,
            idx: int = 0,
    ):
        super().__init__()
        assert poolsize >= 1, "pool_size must be >= 1"
        assert 1 <= top_k <= poolsize, "top_k must be in [1, pool_size]"
        assert not (cluster == "prototype" and balance_prior_lambda != 0.0), "Cannot use balancing prior with prototype clustering."
        assert z_loss_coef >= 0.0, "z_loss_coef must be non-negative."
        assert aux_loss_coef >= 0.0, "aux_loss_coef must be non-negative."
        assert aux_loss_batch_coef >= 0.0, "aux_loss_batch_coef must be non-negative."
        assert prob_sparse_loss_coef >= 0.0, "prob_sparse_loss_coef must be non-negative."
        assert embedding_channels % len(in_channels) == 0, "embedding_channels must be divisible by number of levels."

        self.eps = eps
        self.balance_prior_lambda = balance_prior_lambda
        self.ema_decay = ema_decay
        self.top_k = top_k
        self.poolsize = poolsize
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_batch_coef = aux_loss_batch_coef
        self.prob_sparse_loss_coef = prob_sparse_loss_coef
        self.n_skip = n_skip
        self.idx = idx
        self.buffer_maxsize = buffer_maxsize
        self.buffer_nskip = buffer_nskip

        self.norm = nn.LayerNorm(in_channels)

        # Buffers for cluster results
        self.buffer = []
        self.register_buffer("skip_cnt", torch.tensor(buffer_nskip, dtype=torch.int64))
        self.register_buffer("buffer_cnt", torch.tensor(buffer_maxsize, dtype=torch.int64))
        self.register_buffer("route_ema", torch.zeros((self.poolsize,)))
        self.register_buffer("centers", torch.empty(in_channels, poolsize))
        self.register_buffer("cluster_size", torch.zeros(poolsize))
        self.register_buffer("embed_avg", torch.empty(in_channels, poolsize))
        self.register_buffer("is_fitted", torch.tensor(0, dtype=bool))


    def _balancing_prior(self) -> Tensor:
        """
        Build additive log-prior for logits from EMA usage:
        log_prior_i = -log(usage_i + eps), scaled by balance_lambda.
        """
        # Normalize EMA to a rate-like vector (sum to 1) when non-zero
        usage = self.counter.clamp_min(self.eps)
        usage = usage / usage.sum().clamp_min(self.eps)  # make it comparable across batch sizes
        log_prior = -torch.log(usage)                    # higher for rarely used experts
        return self.balance_prior_lambda * log_prior           # [pool_size]


    def solve_route(self, logits: Tensor) -> Tensor:
        logits_topk = logits.clone()

        top_k = torch.topk(logits_topk, k=self.top_k, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(logits_topk, dtype=torch.bool).scatter_(dim=-1, index=top_k.indices, value=True)  # [B, pool_size]

        masked_logits = torch.where(topk_mask, logits, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
        weights = torch.softmax(masked_logits.float(), dim=-1).to(logits.dtype)  # [B, pool_size]

        # Update frequency stats
        if self.training and self.poolsize > 1:
            with torch.no_grad():
                batch_select = topk_mask.to(logits.dtype).sum(dim=0) # [pool_size], how many times each expert chosen
                dist_fn.all_reduce(batch_select)
                self.route_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_select) # EMA update

                # Logging
                focus_score = self.focus_metric(self.route_ema)
                register_extra_metric(self, f"route_focus{self.idx}", focus_score, op="mean")

        return weights, topk_mask, top_k.indices

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self._balancing_prior().unsqueeze(0).to(x.device, x.dtype)

        # Decrement skip counter regardless of mode to avoid being stuck in eval
        if self.skip_cnt > 0:
            if self.training:
                self.skip_cnt.sub_(1)
            return self.solve_route(logits)

        x = x.detach()
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.norm(x)

        if self.is_fitted:
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ self.centers
                + self.centers.pow(2).sum(0, keepdim=True)
            )

            if self.training:
                with torch.no_grad():
                    # ema update top centers
                    _, embed_ind = (-dist).max(1)
                    embed_onehot = F.one_hot(embed_ind, self.poolsize).type(x.dtype)
                    embed_onehot_sum = embed_onehot.sum(0)
                    embed_sum = x.transpose(0, 1) @ embed_onehot

                    dist_fn.all_reduce(embed_onehot_sum)
                    dist_fn.all_reduce(embed_sum)

                    self.cluster_size.data.mul_(self.ema_decay).add_(
                        embed_onehot_sum, alpha=1 - self.ema_decay
                    )
                    self.embed_avg.data.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
                    n = self.cluster_size.sum()
                    cluster_size = (
                        (self.cluster_size + self.eps) / (n + self.poolsize * self.eps) * n
                    )
                    embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)

                    self.centers.data.copy_(embed_normalized)

            logits = logits - dist
            return self.solve_route(logits)
        
        if self.buffer_cnt > 0 and self.training:
            self.buffer.append(x.detach())
            self.buffer_cnt.sub_(1)

        if self.buffer_cnt == 0 and self.training:
            with torch.no_grad():
                local_buffer = torch.cat(self.buffer, dim=0)
                global_buffer = dist_fn.all_gather(local_buffer)
                global_buffer = torch.cat(global_buffer, dim=0)

                global_buffer = global_buffer.unsqueeze(0)

                self.cluster.fit(global_buffer)

                if self.cluster.is_fitted: # Sync kernel state
                    result: ClusterResult = self.cluster._result

                    self.centers.data.copy_(result.centers.squeeze(0).transpose(0,1))
                    self.embed_avg.data.copy_(self.centers)

                    self.is_fitted.fill_(1)

                else:
                    print("KmeansNav2d cluster fitting failed.")
                    self.buffer.clear()
                    self.buffer_cnt.fill_(self.buffer_maxsize)
                
                self.buffer.clear()
        
        return self.solve_route(logits)