# -*- coding: utf-8 -*-
# Print model FLOPs during sanity checking, once, using fvcore.
# Works with Lightning >=2.x (and 1.x fallback). Prints only on rank-0.

from typing import Any, Callable
import math
import torch
import lightning as L 
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from fvcore.nn import FlopCountAnalysis, flop_count_table
import contextlib

PARAMETER_NUM_UNITS = [" ", "K", "M", "G", "T"]

def get_human_readable_count(number: int) -> str:
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(math.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


class CmrInputsMapper:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, batch):
        return tuple(batch[k] for k in self.keys)

class FlopsProfiler(Callback):
    def __init__(
        self,
        inputs_mapper: Callable[[Any], Any],
        rich: bool = False,
    ):
        super().__init__()
        self.rich = rich
        self.inputs_mapper = inputs_mapper
        self._flops_sanity_checked = False

    def _should_run(self, trainer: L.Trainer) -> bool:
        if self._flops_sanity_checked:
            return False
        if not getattr(trainer, "sanity_checking", False):
            return False
        return True
    
    @rank_zero_only
    def _print_header(self, total_flops: int):
        human_readable_flops = get_human_readable_count(total_flops).replace("B", "G")
        print(f"\n[FLOPs] Total FLOPs: {total_flops} ({human_readable_flops})\n")

    def on_validation_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Trigger once at the first sanity val batch
        if not self._should_run(trainer):
            return

        # Map batch -> model inputs
        model_inputs = self.inputs_mapper(batch)

        # Compute FLOPs with fvcore
        was_training = pl_module.training
        pl_module.eval()
        save_path = trainer.logger.save_dir + "/flops.txt"
        with torch.no_grad():
            with contextlib.ExitStack() as stack:
                f = stack.enter_context(open(save_path, "w", encoding="utf-8")) if trainer.is_global_zero else stack.enter_context(contextlib.nullcontext())
                stack.enter_context(contextlib.redirect_stdout(f))
                stack.enter_context(contextlib.redirect_stderr(f))
                fca = FlopCountAnalysis(pl_module, model_inputs)
                total_flops = fca.total()
            if trainer.is_global_zero:
                print(f"FLOPs report saved to {save_path}")
            
        if trainer.is_global_zero:
            if self.rich:
                table = flop_count_table(fca)
                print(table)
            self._print_header(total_flops)

        # Restore mode and mark done
        pl_module.train(was_training)
        self._flops_sanity_checked = True

    # Reset the flag after sanity checking completes (so normal fit proceeds as usual)
    def on_sanity_check_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._flops_sanity_checked = False