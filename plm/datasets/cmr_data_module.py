import pytorch_lightning as pl
from utils.naneu.common import importlib
from os import PathLike
from typing import List, Literal
from utils.naneu.common.importlib import LazyModule
from datasets import VolumeSampler
import torch

def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    base_seed = worker_info.seed

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seed= base_seed + torch.distributed.get_rank() * worker_info.num_workers
    else:
        seed = base_seed

    seed = seed % (2**32)
    dataset.set_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)

class CmrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: PathLike | str,
        val_paths: List[PathLike | str],
        dataset_class: str,
        dataset_init_args: dict,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.dataset_class = importlib.LazyModule(dataset_class)
        self.dataset_init_args = dataset_init_args
        self.train_path = train_path
        self.val_paths = val_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(
        self,
        path: PathLike | str,
        dataset_class,
        dataset_init_args: dict,
        task: Literal["train", "val"] = "train",
    ) -> torch.utils.data.DataLoader:
        dataset_init_args = dataset_init_args.copy()
        for key, value in dataset_init_args.items():
            if isinstance(value, dict) and 'class_path' in value:
                dataset_init_args[key] = LazyModule(value['class_path'])(**value.get('init_args', {}))

        if ("train" not in task) and ("balance_sampler" in dataset_init_args):
            dataset_init_args["balance_sampler"] = None

        dataset = dataset_class(
            path=path,
            **dataset_init_args,
        )

        # Setting the distributed sampler if available
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if "train" in task:
                sampler = torch.utils.data.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True,
                )
            else:
                sampler = VolumeSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=False,
                )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=True if "train" in task and sampler is None else False,
            pin_memory=True if torch.cuda.is_available() and "train" in task else False,
        )
        return dataloader

    def set_logger(self, logger):
        self.logger = logger

    def train_dataloader(self):
        return self._create_data_loader(self.train_path, self.dataset_class, self.dataset_init_args, task="train")
    
    def val_dataloader(self):
        return [self._create_data_loader(path, self.dataset_class, self.dataset_init_args, task = "val") for path in self.val_paths]
    

class CmrInferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        src_path: PathLike | str,
        dataset_class: str,
        dataset_init_args: dict,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.dataset_class = importlib.LazyModule(dataset_class)
        self.dataset_init_args = dataset_init_args
        self.src_path = src_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(
        self,
        src_path: PathLike | str,
        dataset_class,
        dataset_init_args: dict,
    ) -> torch.utils.data.DataLoader:
        dataset_init_args = dataset_init_args.copy()
        for key, value in dataset_init_args.items():
            if isinstance(value, dict) and 'class_path' in value:
                dataset_init_args[key] = LazyModule(value['class_path'])(**value.get('init_args', {}))

        dataset = dataset_class(
            path=src_path,
            **dataset_init_args,
        )

        # Setting the distributed sampler if available
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = VolumeSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=False
        )
        return dataloader

    def set_logger(self, logger):
        self.logger = logger

    def predict_dataloader(self):
        return self._create_data_loader(self.src_path, self.dataset_class, self.dataset_init_args)
    
   