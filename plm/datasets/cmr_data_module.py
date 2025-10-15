import lightning.pytorch as pl
from os import PathLike
from typing import List, Literal
from utils.naneu.common.importlib import LazyModule
from datasets import VolumeSampler
from dataclasses import dataclass
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

@dataclass
class DatasetConfig:
    path: PathLike | str
    dataset_class: str
    dataset_init_args: dict

class CmrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: DatasetConfig,
        val_configs: List[DatasetConfig],
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_configs = val_configs
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dataset placeholders
        self.trainset: torch.utils.data.Dataset = None
        self.valsets: List[torch.utils.data.Dataset] = None
    
    def setup(self, stage: str):
        self.trainset = self._create_dataset(self.train_config)
        self.valsets = [self._create_dataset(config) for config in self.val_configs]

    def _create_dataset(self, config: DatasetConfig):
        path = config.path
        dataset_class = config.dataset_class
        dataset_init_args = config.dataset_init_args.copy()
        for key, value in dataset_init_args.items():
            if isinstance(value, dict) and 'class_path' in value:
                dataset_init_args[key] = LazyModule(value['class_path'])(**value.get('init_args', {}))

        dataset = LazyModule(dataset_class)(
            path=path,
            **dataset_init_args,
        )
        return dataset

    def _create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        task: Literal["train", "val"] = "train",
    ) -> torch.utils.data.DataLoader:
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
        return self._create_dataloader(self.trainset, task = "train")
    
    def val_dataloader(self):
        return [self._create_dataloader(dataset, task = "val") for dataset in self.valsets]
    

class CmrInferenceDataModule(CmrDataModule):
    def __init__(
        self,
        config: DatasetConfig,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dataset placeholders
        self.dataset: torch.utils.data.Dataset = None

    def setup(self, stage: str):
        self.dataset = self._create_dataset(self.config)

    def _create_data_loader(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> torch.utils.data.DataLoader:
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
        return self._create_data_loader(self.dataset)
    
   