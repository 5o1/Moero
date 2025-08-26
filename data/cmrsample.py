import torch
from typing import List, Dict
from dataclasses import dataclass

class CmrSliceSample:
    def __init__(self, data: torch.Tensor, seq_shape, seq_idx):
        self.data = data        # Data for the single slice
        self.seq_shape = seq_shape  # Original volume shape
        self.seq_idx = (seq_idx,) if isinstance(seq_idx, int) else seq_idx          # Index of the current slice
        
        self.nseqdim = len(self.seq_idx) # number of dims like time, slice ...

    def __repr__(self):
        return f"Cmr4dSliceSample(from seq sample:{self.seq_shape}, idx: {self.seq_idx}, datashape: {self.data.shape})"


class CmrVolumeSample:
    def __init__(self, data: torch.Tensor):
        self.data = data

    def resample(self, idx) -> CmrSliceSample:
        """Extract a single slice from the volume"""
        slice_data = self.data[idx]
        raw_shape = self.data.shape
        return CmrSliceSample(slice_data, raw_shape, idx)

    @staticmethod
    def restore(slice_samples: List[CmrSliceSample]) -> "CmrVolumeSample":
        """
        Combine multiple Cmr4dSliceSample objects into a single Cmr4dVolumeSample.
        """
        # Sort the slices by their index (to ensure correct order)
        slice_samples = sorted(slice_samples, key=lambda x: x.seq_idx)

        # Validate that all slices have consistent raw_shape
        raw_shapes = {tuple(sample.raw_shape) for sample in slice_samples}
        if len(raw_shapes) > 1:
            raise ValueError("All slices must have the same raw_shape.")
        
        # Extract data from slices and stack them
        stacked_data = torch.stack([sample.data for sample in slice_samples], dim=0)
        
        # Create and return the volume sample
        return CmrVolumeSample(stacked_data)

    def __repr__(self):
        return f"Cmr4dVolumeSample(shape={self.data.shape})"

class CmrSampleBase(dict):
    def __post_init__(self):
        values = list(self.values())
        if len(values) > 1 and isinstance(values[0], dict) and all(value is None for value in values[1:]):
            for key, value in values[0].items():
                self[key] = value

    def update(self, d: dict):
        for key, value in d.items():
            self[key] = value
    
    def precision(self, precision: int):
        """
        Convert all torch.Tensor attributes to the specified precision.
        """
        d = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                dtype = value.dtype
                if torch.is_complex(value):
                    d[key] = value.to(getattr(torch, f"complex{precision*2}"))
                elif torch.is_floating_point(value):
                    d[key] = value.to(getattr(torch, f"float{precision}"))
                elif "int" in str(dtype):
                    d[key] = value.to(getattr(torch, f"int{precision}"))
                else:
                    d[key] = value
            else:
                d[key] = value
        return self.__class__(**d)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def to(self, *args, **kwargs):
        d = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.to(*args, **kwargs)
            else:
                d[key] = value
        return self.__class__(**d)
    

@dataclass
class Cmr25Sample(CmrSampleBase):
    masked_kspace: torch.Tensor = None
    mask: torch.Tensor = None
    mask_type: str = None
    target: torch.Tensor = None
    datarange: torch.Tensor = None
    fname: str = None
    seqidx: torch.Tensor = None
    seqshape: torch.Tensor = None

@dataclass
class Cmr25ValidationOutputSample(CmrSampleBase):
    img_pred: torch.Tensor = None # b t s c h w
    img_zf: torch.Tensor = None
    csm: torch.Tensor = None
    datarange: torch.Tensor = None
    loss: torch.Tensor = None
    batch_idx: int = None
    dataloader_idx: int = None

    # def __post_init__(self):
    #     super().__post_init__()
    #     self.img_pred = self.img_pred.detach().to('cpu')
    #     self.img_zf = self.img_zf.detach().to('cpu')
    #     self.csm = self.csm.detach().to('cpu')
    #     self.datarange = self.datarange.detach().to('cpu')
    #     self.loss = self.loss.detach().to('cpu')
    #     self.batch_idx = int(self.batch_idx)
    #     self.dataloader_idx = int(self.dataloader_idx)


@dataclass
class Cmr25InferenceOutputSample(CmrSampleBase):
    img_pred: torch.Tensor = None # b t s c h w
    img_zf: torch.Tensor = None
    csm: torch.Tensor = None
    debug_terms: Dict[str, torch.Tensor] = None
    fname: str = None
    seqidx: torch.Tensor = None
    seqshape: torch.Tensor = None