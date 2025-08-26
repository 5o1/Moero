import torch
from os import PathLike
import os
from typing import Tuple, List, Literal, Sequence, Callable
from data.cmrsample import Cmr25Sample
import h5py
import math
import numpy as np
from glob import glob
from einops import rearrange
from data.transforms.maskgenerator import UniformMaskGenerator, KtGaussianMaskGenerator, KtRadialMaskGenerator


class MixedRandomMaskGenerator(torch.nn.Module):
    def __init__(
            self,
            acc_factors: List[int] = [2, 4, 8, 12, 16, 20, 24],
            n_calibs: List[int] = [16, 20],
    ):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(torch.initial_seed())
        self.maskgen_pool = [
            UniformMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
            KtGaussianMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
            KtRadialMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
        ]
        self.maskgen_weights = torch.as_tensor([
            1,
            1,
            1,
        ], dtype=torch.float32)
        self.masktype_pool = [
            "Uniform",
            "ktGaussian",
            "ktRadial"
        ]
    def set_seed(self, seed: int):
        self.rng.manual_seed(seed)

    def forward(self, size: Sequence[int]) -> torch.Tensor:
        idx = torch.multinomial(self.maskgen_weights, num_samples=1, generator = self.rng).item()
        maskgen = self.maskgen_pool[idx]
        masktype = self.masktype_pool[idx]

        mask, accel_factor, ncalib = maskgen(size)
        masktype = self.masktype_pool[idx] + str(accel_factor)
        return mask, masktype


class Cmr25TrainingTransform(torch.nn.Module):
    def __init__(
            self,
            acc_factors: List[int] = [2, 4, 8, 12, 16, 20, 24],
            n_calibs: List[int] = [16, 20],
            ):
        super().__init__()
        self.maskgen = MixedRandomMaskGenerator(acc_factors, n_calibs)

    def set_seed(self, seed: int):
        self.maskgen.set_seed(seed)
    
    def forward(self, sample: Cmr25Sample) -> Cmr25Sample:
        size = (sample.masked_kspace.size(0), sample.masked_kspace.size(-2), sample.masked_kspace.size(-1))
        mask, masktype = self.maskgen(size) # (batch, phase, readout)

        mask = rearrange(mask, "t readout phase -> t 1 1 readout phase")  # Add channel dimension

        sample.masked_kspace = sample.masked_kspace * mask # Apply mask to k-space data
        sample.mask = mask
        sample.mask_type = masktype
        return sample

class Cmr25TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: PathLike | str,
        transform: torch.nn.Module = None,
        n_adj_frame: int = 5,
        n_adj_slice: int = 1,
        which_adj: Literal["frame", "slice"] = "frame",
        balance_sampler: Callable = None,
    ):
        if not isinstance(path, (PathLike, str)):
            raise TypeError(f"Expected path to be of type PathLike, got {type(path)}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Path {path} is not a directory.")

        self.path = path
        self.n_adj_frame = n_adj_frame
        self.n_adj_slice = n_adj_slice
        self.which_adj = which_adj

        self.transform = transform

        # self.collate_fn = _collate_fn

        # get all the kspace mat files from root, under folder or its subfolders
        self.fnamelist = sorted(list(glob(f"{path}/*.h5")))
        print(f'number of paths `{path}`: {len(self.fnamelist)}')
        if balance_sampler:
            self.fnamelist = balance_sampler(self.fnamelist)
            print(f"Balance sampler is available, number of paths after balance: {len(self.fnamelist)}")

        assert n_adj_frame % 2 == 1 and n_adj_slice % 2 == 1, "Number of adjacent frames and slices must be odd."

        self.raw_samples = []

        for fname in self.fnamelist:
            with h5py.File(fname, 'r') as hf:
                attrs = dict(hf.attrs)
                shape = torch.as_tensor(attrs['shape'])

                if len(shape) == 5:
                    seqshape = shape[:2]
                    for ti in range(seqshape[0]):
                        for zi in range(seqshape[1]):
                            seqidx = torch.as_tensor((ti, zi))
                            self.raw_samples.append((fname, seqidx, seqshape))
                elif len(shape) == 4:
                    seqshape = shape[:1]
                    for zi in range(shape[0]):
                        seqidx = torch.as_tensor((zi,))
                        self.raw_samples.append((fname, seqidx, seqshape))
                else:
                    raise ValueError(f"Unsupported data formats: {fname} with shape {shape}")

    def _get_indices(self, idx: int, length: int, target_length: int, pad: Literal["circle", "clamp", "mirror", False] = "clamp") -> List[int]:
        if not 0 <= idx < length:
            raise ValueError(f"Invalid index {idx} for volume with only one time point.")

        start = idx + math.ceil(-target_length / 2)
        end = idx + math.ceil(target_length / 2)
        indices = torch.arange(start, end, dtype = torch.int64)
        if pad == "circle":
            indices = (indices + length) % length 
        elif pad == "clamp":
            indices = torch.clamp(indices, 0, length - 1)
        elif pad == "mirror":
            indices = indices.clone()
            low_mask = indices < 0
            high_mask = indices >= length

            indices[low_mask] = -indices[low_mask]
            indices[high_mask] = 2 * length - indices[high_mask] - 2
        elif not pad:
            indices = indices[(indices >= 0) & (indices < length)]
        else:
            raise ValueError(f"Unsupported padding type: {pad}")
        indices = indices.tolist()
        return indices
    
    def __len__(self):
        return len(self.raw_samples)
    
    def set_seed(self, seed: int):
        if self.transform is not None:
            self.transform.set_seed(seed)
    
    def np_getitem_complex_batch(self, data: np.ndarray, *dim_indices: Sequence) -> dict:
        dim_indices = [indices.flatten().tolist() if isinstance(indices, np.ndarray) else indices for indices in dim_indices]
        seq_indices = list(zip(*dim_indices))
        seq_indices_increasing = sorted(list(set(seq_indices)))

        mapping = {seq_idx: i for i, seq_idx in enumerate(seq_indices_increasing)}

        data = [self.np_getitem_complex(data, *seq_idx) for seq_idx in seq_indices_increasing]
        data = np.stack([data[mapping[seq_idx]] for seq_idx in seq_indices], axis=0)
        return data
    
    def np_getitem_complex(self, data: np.ndarray, *dim_indices: Sequence[int] | int) -> np.ndarray:
        if data.dtype == np.dtype([('real', '<f8'), ('imag', '<f8')]):
            slicedata = data[dim_indices]
            return slicedata['real'] + 1j * slicedata['imag']
        else:
            return data[dim_indices]

    def _check_data(self, data: np.ndarray, indices: Tuple[np.ndarray, np.ndarray], fname: str):
        if data.size == 0:
            raise ValueError(
                f"Empty k-space data detected for file '{fname}' with indices {indices}. "
                f"Data shape: {data.shape}."
            )

        if np.isnan(data).any():
            raise ValueError(
                f"k-space data contains NaN values for file '{fname}' with indices {indices}. "
                f"Data shape: {data.shape}."
            )


    def __getitem__(self, idx: int):
        fname, seqidx, seqshape = self.raw_samples[idx]

        with h5py.File(fname, 'r') as hf:
            attrs = dict(hf.attrs)
            kspace = hf["kspace"] # frame slice coil height weight
            rss = hf["reconstruction_rss"] # frame slice height weight
            datarange = torch.as_tensor(attrs["max"])
            
            if len(seqidx) == 2: # 4d data
                ti, zi = seqidx
                nframe, nslice = seqshape[0], seqshape[1]

                n_adj_frame = self.n_adj_frame if self.which_adj == "frame" else min(self.n_adj_frame, nframe) - (min(self.n_adj_frame, nframe) % 2 == 0)
                n_adj_slice = self.n_adj_slice if self.which_adj == "slice" else min(self.n_adj_slice, nslice) - (min(self.n_adj_slice, nslice) % 2 == 0)

                # Make fixed-length adjoint slice or frame indices.
                adj_tis = self._get_indices(ti, nframe, n_adj_frame, pad="clamp")
                adj_sis = self._get_indices(zi, nslice, n_adj_slice, pad="clamp")

                grid_t, grid_s = np.meshgrid(adj_tis, adj_sis, indexing="ij")  # [len(adj_tis), len(adj_sis)]
                grid_t = grid_t.ravel()
                grid_s = grid_s.ravel()

                kdata = self.np_getitem_complex_batch(kspace, grid_t, grid_s)
                self._check_data(kdata, (adj_tis, adj_sis), fname)

                kdata = rearrange(kdata, "(t s) c h w -> t s c h w", t = len(adj_tis), s = len(adj_sis))
                kdata = torch.as_tensor(kdata)

                rss = self.np_getitem_complex(rss, ti, zi)
                self._check_data(rss, (ti, zi), fname)
                rss = torch.as_tensor(rss)
                if rss.ndim != 2:
                    raise ValueError("Ndim of RSS img must be 2.")
                rss = rss.unsqueeze(0) # 1 h w
                
            elif len(seqidx) == 1: # 3d data w/o frames
                zi = seqidx[0]
                nslice = seqshape[0]

                n_adj_frame = self.n_adj_frame if self.which_adj == "frame" else 1
                n_adj_slice = self.n_adj_slice if self.which_adj == "slice" else min(self.n_adj_slice, nslice) - (min(self.n_adj_slice, nslice) % 2 == 0)

                adj_sis = self._get_indices(zi, nslice, self.n_adj_slice, pad="clamp")

                kdata = self.np_getitem_complex_batch(kspace, adj_sis)
                self._check_data(kdata, adj_sis, fname)
                
                kdata = rearrange(kdata, "s c h w -> 1 s c h w", s = len(adj_sis))
                kdata = torch.as_tensor(kdata)
                kdata = kdata.expand((n_adj_frame, -1, -1, -1, -1))  # Expand to match n_adj_frame
                
                rss = self.np_getitem_complex(rss, zi)
                self._check_data(rss, adj_sis, fname)
                rss = torch.as_tensor(rss)
                if rss.ndim != 2:
                    raise ValueError("Ndim of RSS img must be 2.")
                rss = rss.unsqueeze(0) # 1 h w
            else:
                raise ValueError(f"Unsupported idx formats: {fname} with sliceidx {seqidx}")
            
        sample = Cmr25Sample(
            masked_kspace=kdata,
            mask = None,
            mask_type= None,
            target = rss,
            datarange=datarange,
            fname = fname,
            seqidx = seqidx,
            seqshape = seqshape,
        )

        if self.transform is not None:
            with torch.no_grad():
                sample = self.transform(sample)

        if self.which_adj == "frame":  # transpose t s
            sample.masked_kspace = rearrange(sample.masked_kspace, "t s c h w -> s t c h w")
            sample.mask = rearrange(sample.mask, "t s c h w -> s t c h w")
        return sample.precision(32)
