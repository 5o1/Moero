from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _enable_get_lr_call, _warn_get_lr_called_within_step
import math
from copy import deepcopy


class DecayCosineAnnealingWarmRestarts(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        eta_min_0: float,
        T_mult: int | float = 1,
        eta_min_min: float | None = None,
        lr_max_mult: int | float = 0.666666,
        lr_min_mult: int | float = 1,
        last_epoch: int = -1,
    ):  # noqa: D107
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1:
            raise ValueError(f"Expected T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min_0, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min_0} of type {type(eta_min_0)}"
            )
        if lr_max_mult <= 0 or not isinstance(lr_max_mult, (float, int)):
            raise ValueError(f"Expected positive float lr_max_mult, but got {lr_max_mult}")
        if lr_min_mult <= 0 or not isinstance(lr_min_mult, (float, int)):
            raise ValueError(f"Expected positive float lr_min_mult, but got {lr_min_mult}")
        if lr_min_mult != 1:
            if eta_min_min is None:
                raise ValueError(
                    "Expected eta_min_0 to be provided when lr_min_mult is not 1."
                )
            elif not eta_min_0 >= eta_min_min:
                raise ValueError(
                    f"Expected eta_min_0 > eta_min_min, but got eta_min_0={eta_min_0} and eta_min_min={eta_min_min}"
                )

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult_base = T_mult
        self.eta_min_base = eta_min_0
        self.eta_min_min = eta_min_min if eta_min_min is not None else eta_min_0
        self.eta_max_base = [group["lr"] for group in optimizer.param_groups]
        self.lr_max_mult = lr_max_mult
        self.lr_min_mult = lr_min_mult
        self.T_cur = last_epoch

        self.T_mult = self.T_mult_base
        self.eta_min = deepcopy(self.eta_min_base)
        self.eta_max = deepcopy(self.eta_max_base)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the updated learning rate."""
        _warn_get_lr_called_within_step(self)

        return [
            self.eta_min
            + (eta_max - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for eta_max in self.eta_max
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update."""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = int(self.T_i * self.T_mult)

                # Apply decay to eta_max and eta_min after restart
                self.eta_min = max(self.eta_min_min, self.eta_min * self.lr_min_mult)
                self.eta_max = [self.eta_min + (lr - self.eta_min) * self.lr_max_mult for lr in self.eta_max]
                
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = epoch // self.T_0
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)

                # Apply decay to eta_max and eta_min after restart
                self.eta_min = max(self.eta_min_min, self.eta_min_base * self.lr_min_mult ** n)
                self.eta_max = [self.eta_min + (lr_max - self.eta_min) * (self.lr_max_mult) ** n for lr_max in self.eta_max_base]

            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]