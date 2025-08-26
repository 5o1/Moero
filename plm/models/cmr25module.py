import torch
from typing import Literal
from data.transforms.crop import center_crop_to_smallest
from data.cmrsample import Cmr25Sample, Cmr25ValidationOutputSample, Cmr25InferenceOutputSample
from .mrimodule import MriModule
from utils.naneu.nn.modules.loss import SSIMLoss, VGGLoss
from utils.naneu.common.importlib import LazyModule
from utils.naneu.helpers.context import ExtraContext
from utils.naneu import fft

class Cmr25Module(MriModule):
    loss_fn: torch.nn.Module
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_class_path: str = "torch.optim.AdamW",
        optimizer_init_kwargs: dict = {"lr": 0.0002, "weight_decay": 0.01},
        scheduler_class_path: str | None = None,
        scheduler_init_kwargs: dict = {},
        scheduler_interval: Literal["epoch", "step"] = "step",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.loss_fn = SSIMLoss(7, kernel="gauss")

        if self.fine_tuning:
            self.perceptual_fn = VGGLoss()

    def configure_optimizers(self):
        optim = LazyModule(self.hparams.optimizer_class_path)(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.hparams.optimizer_init_kwargs
        )
        if self.hparams.scheduler_class_path is None:
            return optim

        # step lr scheduler
        scheduler = LazyModule(self.hparams.scheduler_class_path)(
            optim,
            **self.hparams.scheduler_init_kwargs
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.hparams.scheduler_interval,
        }
        return [optim], [scheduler_config]
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
        output_dict = self.model(masked_kspace, mask)
        return output_dict

    def training_step(self, batch: Cmr25Sample, batch_idx: int):
        if not torch.isfinite(batch.masked_kspace).all():
            raise ValueError(f"Invalid masked_kspace in batch {batch_idx}")

        with ExtraContext(self) as ctx:
            output_dict = self(batch.masked_kspace, batch.mask)
            output = output_dict['img_pred']
            target, output = center_crop_to_smallest(
                batch.target, output)
            
            loss = self.loss_fn(
                output, target, batch.datarange
            )

            self.log("train_loss", loss.detach(), prog_bar=True)

            total_loss = loss
            if len(ctx_loss_dict:= ctx.get_losses()) > 0:
                for loss_name, loss_score in ctx_loss_dict.items():
                    self.log(f"train_{loss_name}", loss_score.detach(), prog_bar=True)
                    total_loss += loss_score

            if self.fine_tuning:
                vgg_loss = self.perceptual_fn(output, target)
                total_loss += vgg_loss
                self.log("train_vgg_loss", vgg_loss.detach(), prog_bar=True)

            self.log("train_total_loss", total_loss.detach(), prog_bar=True)

            if len(ctx_metric_dict:= ctx.get_metrics()) > 0:
                for metric_name, metric_score in ctx_metric_dict.items():
                    self.log(f"train_{metric_name}", metric_score.detach(), prog_bar=True)

        return total_loss

    def validation_step(self, batch: Cmr25Sample, batch_idx: int, dataloader_idx: int = 0):
        if not torch.isfinite(batch.masked_kspace).all():
            raise ValueError(f"Invalid masked_kspace in batch {batch_idx}")
        
        with ExtraContext(self) as ctx:
            output_dict = self(batch.masked_kspace, batch.mask)
            output = output_dict['img_pred']
            img_zf = output_dict['img_zf']
            csm = output_dict['csm']
            target, output = center_crop_to_smallest(
                batch.target, output)
            _, img_zf = center_crop_to_smallest(
                batch.target, img_zf)

            datarange = batch.datarange
            val_loss = self.loss_fn(
                    output, target, datarange
                )

            if len(ctx_loss_dict:= ctx.get_losses()) > 0:
                for loss_name, loss_score in ctx_loss_dict.items():
                    self.log(f"val_{loss_name}", loss_score.detach(), prog_bar=True, sync_dist=True)
            
            if len(ctx_metric_dict:= ctx.get_metrics()) > 0:
                for metric_name, metric_score in ctx_metric_dict.items():
                    self.log(f"val_{metric_name}", metric_score.detach(), prog_bar=True, sync_dist=True)

        return Cmr25ValidationOutputSample(
            img_pred=output,
            img_zf=img_zf,
            csm=csm,
            datarange=datarange,
            loss = val_loss,
            batch_idx=batch_idx,
            dataloader_idx = dataloader_idx,
        )

    def predict_step(self, batch: Cmr25Sample, batch_idx: int, dataloader_idx=0):
        if not torch.isfinite(batch.masked_kspace).all():
            raise ValueError(f"Invalid masked_kspace in batch {batch_idx}")
        
        # Validation set contains corrupted data
        slicedata = batch.masked_kspace
        slicedata = slicedata[0, slicedata.size(1) // 2, slicedata.size(2) // 2]
        if (slicedata == 0).all() or not slicedata.isfinite().all():
            print(f"Zero value input data detected in batch {batch_idx}, fname={batch.fname}, seqidx={batch.seqidx}, seqshape={batch.seqshape}")
            return Cmr25InferenceOutputSample(
                img_pred=torch.zeros_like(batch.target),
                img_zf=torch.zeros_like(batch.target),
                csm=torch.zeros_like(batch.target).expand(-1, batch.masked_kspace.size(-3), -1, -1),
                debug_terms={},
                fname=batch.fname,
                seqidx=batch.seqidx,
                seqshape=batch.seqshape
            )
        
        with ExtraContext(self) as ctx:
            output_dict = self(batch.masked_kspace, batch.mask)
            output = output_dict['img_pred']
            img_zf = output_dict['img_zf']
            csm = output_dict['csm']
            debug_terms = ctx.get_outputs()

            if output.isnan().any():
                print(f"NaN detected in output at fname{batch.fname}, seqidx={batch.seqidx}, seqshape={batch.seqshape}")
            
        return Cmr25InferenceOutputSample(
            img_pred=output,
            img_zf=img_zf,
            csm=csm,
            debug_terms=debug_terms,
            fname=batch.fname,
            seqidx=batch.seqidx,
            seqshape=batch.seqshape
        )
