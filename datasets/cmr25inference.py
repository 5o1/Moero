import torch
from typing import Literal
import h5py
import numpy as np
from einops import rearrange
from os import PathLike
from data.cmrsample import Cmr25Sample

from . import Cmr25TrainingDataset

class Cmr25InferenceDataset(Cmr25TrainingDataset):
    def __init__(
        self,
        path: PathLike | str,
        transform: torch.nn.Module = None,
        n_adj_frame: int = 3,
        n_adj_slice: int = 5,
        which_adj: Literal["frame", "slice"] = "frame",
    ):
        super().__init__(
            path = path,
            transform=transform,
            n_adj_frame=n_adj_frame,
            n_adj_slice=n_adj_slice,
            which_adj=which_adj,
            balance_sampler=None
        )

    def __getitem__(self, idx: int):
        fname, seqidx, seqshape = self.raw_samples[idx]

        with h5py.File(fname, 'r') as hf:
            attrs = dict(hf.attrs)
            kus = hf["kus"]
            mask = hf["mask"]
            
            if len(seqidx) == 2:
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

                kdata = self.np_getitem_complex_batch(kus, grid_t, grid_s)
                self._check_data(kdata, (adj_tis, adj_sis), fname)

                kdata = rearrange(kdata, "(t s) c h w -> t s c h w", t = len(adj_tis), s = len(adj_sis))
                kdata = torch.as_tensor(kdata)

                mask_data = self.np_getitem_complex_batch(mask, adj_tis)
                mask_data = rearrange(mask_data, "t h w -> t 1 1 h w", t = len(adj_tis))
                mask_data = torch.as_tensor(mask_data).float()

            elif len(seqidx) == 1:
                zi = seqidx[0]
                nslice = seqshape[0]

                n_adj_frame = self.n_adj_frame if self.which_adj == "frame" else 1
                n_adj_slice = self.n_adj_slice if self.which_adj == "slice" else min(self.n_adj_slice, nslice) - (min(self.n_adj_slice, nslice) % 2 == 0)

                adj_sis = self._get_indices(zi, nslice, self.n_adj_slice, pad="clamp")

                kdata = self.np_getitem_complex_batch(kus, adj_sis)
                self._check_data(kdata, adj_sis, fname)

                kdata = rearrange(kdata, "s c h w -> 1 s c h w", s = len(adj_sis))
                kdata = torch.as_tensor(kdata)
                kdata = kdata.expand((n_adj_frame, -1, -1, -1, -1))  # Expand to match n_adj_frame

                mask_data = rearrange(mask[()], "h w -> 1 1 1 h w")
                mask_data = torch.as_tensor(mask_data).float()

            else:
                raise ValueError(f"Unsupported idx formats: {fname} with sliceidx {seqidx}")

        sample = Cmr25Sample(
            masked_kspace=kdata,
            mask=mask_data,
            mask_type="None",
            target = torch.zeros(kdata.shape[-2:], dtype = kdata.dtype, device=kdata.device).unsqueeze(0).abs(),  # Placeholder target
            datarange = torch.tensor(0),
            fname = fname,
            seqidx=seqidx,
            seqshape = seqshape
        )

        if self.transform is not None:
            with torch.no_grad():
                sample = self.transform(sample)

        if self.which_adj == "frame":  # transpose t s
            sample.masked_kspace = rearrange(sample.masked_kspace, "t s c h w -> s t c h w")
            sample.mask = rearrange(sample.mask, "t s c h w -> s t c h w")

        return sample.precision(32)