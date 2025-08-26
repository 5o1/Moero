"""
Submit Script for Cmr25.

Requirement:

1. The tensor has the shape [[frame] slice c h w], where c = 1. Alternatively, the input can have the shape [[frame] slice c h w] when using the --nochdim option.
2. Input File format is `*.pt`.
2. Input File Name, Only `@` is required to split directory path segmentation. 

Example:
`Mapping@ValidationSet@UnderSample_TaskR1@Center004@Siemens_15T_Aera@P005@T2map_kus_ktGaussian8.pt`
"""

import torch
import argparse
import os
from scipy.io import savemat
from tqdm import tqdm
from glob import glob

from .import mrimodality as mm
from .transform4ranking import transform4ranking as transform

def resolve_savepath(fname: str, src_extend: str = ".pt", tgt_extend: str = ".mat") -> str:
    fname = fname.replace("@", "/")
    fname = fname.replace(src_extend, tgt_extend)
    return fname

def resolve_modality(fname: str) -> mm.MriModality:
    fname = fname.lower()
    modalities_list = sorted([
        mm.BlackBlood,
        mm.T1mappost,
        mm.T2smap,
        mm.T1map,
        mm.T2map,
        mm.Cine,
        mm.LGE,
        mm.Perfusion,
        mm.T1rho,
        mm.Flow2d
    ],key = lambda m:len(m.name), reverse=True) # Long to short

    for modality in modalities_list:
        if modality.name in fname:
            return modality()
    return mm.MriModality()

def save(datasets: dict, fname: str):
    dir_path = os.path.dirname(fname)
    os.makedirs(dir_path, exist_ok=True)

    savemat(fname, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src", help="Path of source .pt folder.")
    parser.add_argument("--tgt", help="Path of target output .png folder.")
    parser.add_argument("--task", help="Task prefix name.")
    parser.add_argument("--nochdim", action="store_true", help="Set nochdim to True if provided, else False")

    args = parser.parse_args()

    src = args.src
    tgt = args.tgt
    taskname = args.task
    nochdim = args.nochdim

    fnamelist = glob(os.path.join(src, "*.pt"))

    for fname in tqdm(fnamelist):
        tensor = torch.load(fname, weights_only=True)

        if not nochdim:
            if tensor.size(-3) != 1:
                raise ValueError(f"Input tensor {fname} must have channel dimension c=1, but got {tensor.size(-3)}")
            tensor = tensor.squeeze(-3) # Remove channel dimension

        modality = resolve_modality(fname)
        tensor = transform(tensor, modality)
        if not torch.isfinite(tensor).all():
            print(f"Sample {fname} may be NaN.")

        prefix = ""
        basename = os.path.basename(fname)
        if "MultiCoil" not in basename:
            prefix = os.path.join("MultiCoil", prefix)
        prefix = os.path.join(taskname, prefix)

        save_to = os.path.join(tgt, prefix, resolve_savepath(basename))
        save({"img4ranking": tensor}, save_to)