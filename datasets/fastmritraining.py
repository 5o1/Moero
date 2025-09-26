import torch
from data.cmrsample import CmrSample
import h5py
import numpy as np
from einops import rearrange

from .cmrdataset import CmrDatasetBase

class FastMriTrainingDataset(CmrDatasetBase):
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
                adj_tis = self._get_indices(ti, nframe, n_adj_frame, pad= self.adj_padding)
                adj_sis = self._get_indices(zi, nslice, n_adj_slice, pad= self.adj_padding)

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

                adj_sis = self._get_indices(zi, nslice, self.n_adj_slice, pad=self.adj_padding)

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
            
        sample = CmrSample(
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
