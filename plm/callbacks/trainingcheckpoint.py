from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import timedelta
import torch
from copy import deepcopy
from typing_extensions import override

class TrainingCheckpoint(ModelCheckpoint):
    def __init__(self, last_n=1, every_n_minites: int = None, every_n_iterations: int =None, last_name: str = 'training-last'):
        monitor = "totalsteps"
        super().__init__(
            filename='training-{epoch}-{step}-{monitor}'.replace("monitor", monitor),
            monitor=monitor,
            mode="max",
            save_last=True,
            save_top_k=last_n,
            every_n_train_steps = every_n_iterations,
            save_weights_only=False,
            train_time_interval=timedelta(minutes=every_n_minites),
            save_on_train_epoch_end = False,
        )
        self.monitor = None
        self.CHECKPOINT_NAME_LAST = last_name

    @override
    def _monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, Tensor]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        monitor_candidates[self.monitor] = monitor_candidates["epoch"] * 1e10 + monitor_candidates["step"]
        return monitor_candidates
    
    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass