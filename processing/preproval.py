"""
Save kus and mask to the same h5 file.

Example of input fname: MultiCoil/Perfusion/ValidationSet/UnderSample_TaskR1/Center001/UIH_30T_umr780/P022/perfusion_kus_Uniform8.mat
Example of output fname: Perfusion@ValidationSet@UnderSample_TaskR1@Center001@UIH_30T_umr780@P022@perfusion_kus_Uniform8.h5

Output keys:
kus: Under Sampled kspase data
mask: corresponding mask
"""
from pqdm.processes import pqdm
from pathlib import Path
from tqdm import tqdm
import re
import h5py
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help = "Path like `/Path/to/TaskR1`")
    parser.add_argument("--tgt", help = "Path like `/Path/to/TaskR1_h5`")
    parser.add_argument('--workers', type=int, default=4, help='number of worker processes for parallel processing')

    args = parser.parse_args()

    mat_folder = Path(args.src)
    save_folder = Path(args.tgt)
    workers = args.workers

    print('matlab data folder: ', mat_folder)
    print('h5 save folder: ', save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    print('## step 1: convert matlab training dataset to h5 dataset')

    filelist = mat_folder.glob("**/*.mat")
    filelist = [str(file) for file in filelist if 'UnderSample' in str(file) and 'Mask' not in str(file)]

    print('number of total matlab files: ', len(filelist))

    def process_file(kdata_path):
        ##* get info from path
        # MultiCoil/Perfusion/ValidationSet/UnderSample_TaskR1/Center001/UIH_30T_umr780/P022/perfusion_kus_Uniform8.mat
        match = re.search(r'MultiCoil/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)', kdata_path)
        modal = match.group(1)
        TrainingSet = match.group(2)
        FullSample = match.group(3)
        center = match.group(4)
        mridevice = match.group(5)
        paid = match.group(6)
        directory, filename = os.path.split(kdata_path)
        fid = os.path.basename(directory)
        ftype = os.path.splitext(filename)[0]
        save_name = f'{modal}@{TrainingSet}@{FullSample}@{center}@{mridevice}@{paid}@{ftype}'
        
        mask_path = kdata_path.replace("UnderSample", "Mask").replace("_kus_", "_mask_")
        save_path = os.path.join(save_folder, save_name + '.h5')

        # load kdata
        with h5py.File(kdata_path, 'r') as f:
            kus = f['kus'][:]

        # load mask
        with h5py.File(mask_path, 'r') as f:
            mask = f['mask'][:]

        ##* swap phase_encoding and readout
        kus = kus.swapaxes(-1,-2)
        mask = mask.swapaxes(-1,-2)
        
        ##* save h5
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('kus', data=kus)
            f.create_dataset('mask', data=mask)

            f.attrs['acquisition'] = modal
            f.attrs['shape'] = kus.shape
            f.attrs['padding_left'] = 0
            f.attrs['padding_right'] = kus.shape[-1]
            f.attrs['encoding_size'] = (kus.shape[-2],kus.shape[-1],1)
            f.attrs['recon_size'] = (kus.shape[-2],kus.shape[-1],1)
            f.attrs['patient_id'] = paid
            f.attrs['center'] = center
            f.attrs['mridevice'] = mridevice
        return save_name

    results = pqdm(filelist, process_file, n_jobs=workers, desc="Processing Files")

    print("Processed files:", results)