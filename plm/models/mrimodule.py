"""
This file includes code modified from the original project found in the
LICENSE file in the root directory of this source tree.

MIT License, Copyright (c) Facebook, Inc. and its affiliates.

PromptMR+ Non-commercial Research License, © 2024 Rutgers, The State University of New Jersey.
---

Change logs:

2025-07-18 5o1

1. Move Tensor to CPU in batch aggregation.
2. Try to aggregate metrics over ranks.
3. Moved prediction code to callbacks.

"""
from collections import defaultdict
from data.cmrsample import Cmr25ValidationOutputSample, Cmr25Sample

import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric
from torch.nn.functional import mse_loss as mse_fn
from utils.naneu.nn.modules import SSIMLoss


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


def normalize_minmax(tensor: torch.Tensor) -> torch.Tensor:
    bias = tensor.min()
    scale = tensor.max() - bias

    tensor = (tensor - bias) / scale
    return tensor, (bias, scale)

class MriModule(pl.LightningModule):
    """
    Abstract super class for deep learning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16, ckpt_path:str = None, ckpt_strict: bool = True, fine_tuning: bool = False, debug = False):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ckpt_strict = ckpt_strict
        self.fine_tuning = fine_tuning # It is recommend turn off `ckpt_strict` during fine tuning because the module may have unused parameters.
        self.debug = debug

        self.val_logs: list = []

        self.num_log_images = num_log_images
        self.val_imagelog_indices = defaultdict(list)

        self.ssim_fn = SSIMLoss(7)

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def log_image(self, key, images, captions):
        self.logger.log_image(key, images, caption=captions, step=self.global_step) # wandb

    def on_validation_batch_end(self, outputs: Cmr25ValidationOutputSample, batch: Cmr25Sample, batch_idx: int, dataloader_idx: int = 0):
        outputs = outputs.to("cpu")
        batch = batch.to("cpu")
        
        # pick a set of images to log if we don't have one already
        if self.val_imagelog_indices.get(dataloader_idx) is None:
            # Determine the number of batches to sample from
            limit_val_batches = self.trainer.limit_val_batches
            if isinstance(limit_val_batches, float) and limit_val_batches <= 1.0:
                num_val_batches = int(limit_val_batches * len(self.trainer.val_dataloaders[dataloader_idx]))
            else:
                num_val_batches = int(limit_val_batches)
            # Randomly sample indices
            self.val_imagelog_indices[dataloader_idx] = torch.randperm(num_val_batches)[:self.num_log_images].tolist()

        # log images
        if batch_idx in self.val_imagelog_indices[dataloader_idx]:
            key = f"val_dataloader{dataloader_idx}_batch{batch_idx}"
            target = batch.target[0].clone() # 1 h w
            img_pred = outputs.img_pred[0].clone() # 1 h w
            img_zf = outputs.img_zf[0].clone() # 1 h w
            mask = batch.mask[0, batch.mask.size(1) // 2, batch.mask.size(2) // 2, 0:1, ...].clone() # 1 h w
            csm = outputs.csm[0,0:1].clone()
            diffrence = torch.abs(target - img_pred)

            img_zf, _ = normalize_minmax(img_zf)
            csm, _ = normalize_minmax(csm)
            img_pred, _ = normalize_minmax(img_pred)
            target, _ = normalize_minmax(target)
            diffrence, _ = normalize_minmax(diffrence)

            self.log_image(
                key,
                [mask, csm, img_zf, img_pred, target, diffrence], 
                captions=['mask', 'csm', 'zf', 'reconstruction', 'target', 'error']
                )

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        dataranges = dict()
        for i, fname in enumerate(batch.fname):
            seq_idx = batch.seqidx[i].clone()
            datarange = outputs.datarange[i].clone()
            img_pred = outputs.img_pred[i].clone()
            target = batch.target[i].clone()

            mse_vals[fname][seq_idx] = mse_fn(img_pred, target).view(1)[0]
            target_norms[fname][seq_idx] = target.norm().view(1)[0]
            ssim_vals[fname][seq_idx] = 1 - self.ssim_fn(target, img_pred, datarange)
            dataranges[fname] = datarange.clone()

        self.val_logs.append({
            "val_loss": outputs.loss.clone(), # issue: https://discuss.pytorch.org/t/pytorch-cannot-allocate-memory/134754/19
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "datarange": dataranges,
        })

    def on_after_backward(self):
        if self.debug:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(name)

    def on_validation_epoch_end(self):        
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        datarange = dict()

        for val_log in self.val_logs:
            losses.append(val_log["val_loss"].view(-1))
            for fname in val_log["mse_vals"].keys():
                mse_vals[fname].update(val_log["mse_vals"][fname])
            for fname in val_log["target_norms"].keys():
                target_norms[fname].update(val_log["target_norms"][fname])
            for fname in val_log["ssim_vals"].keys():
                ssim_vals[fname].update(val_log["ssim_vals"][fname])
            for fname in val_log["datarange"]:
                datarange[fname] = val_log["datarange"][fname]
        self.val_logs.clear()

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == datarange.keys()
        )

        # apply means across image volumes
        local_volumes = len(mse_vals)
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        for fname in mse_vals.keys():
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.as_tensor(
                        datarange[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.as_tensor(local_volumes))
        val_loss = self.ValLoss(torch.cat(losses).sum() if len(losses) > 0 else torch.tensor(0.0))
        tot_slice_examples = self.TotSliceExamples(torch.as_tensor(len(losses), dtype=torch.float))

        # log metrics
        self.log(f"validation_loss", val_loss / tot_slice_examples, prog_bar=True, sync_dist=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, sync_dist=True)


    def setup(self, stage=None):
        if self.ckpt_path:
            is_ddp = torch.distributed.is_initialized()

            rank = torch.distributed.get_rank() if is_ddp else 0

            if rank == 0:
                print(f"Rank {rank}: Loading checkpoint from {self.ckpt_path}")
                checkpoint = torch.load(self.ckpt_path, map_location="cpu")
                state_dict = checkpoint["state_dict"]
                print(f"Rank {rank}: Checkpoint loaded.")

                obj_list = [state_dict]
            else:
                obj_list = [None]

            if is_ddp:
                torch.distributed.broadcast_object_list(obj_list, src=0)

            state_dict = obj_list[0]

            if state_dict:
                self.load_state_dict(state_dict, strict=self.ckpt_strict)
                print(f"Rank {rank}: Model weights loaded successfully!")
            else:
                print(f"Rank {rank}: No state_dict found in the checkpoint.")