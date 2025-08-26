from pqdm.processes import pqdm
import re
import h5py
import os
import argparse
import numpy as np
import torch
from glob import glob
from utils.naneu.fft import ktoi
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for CMRxRecon series dataset')
    parser.add_argument("--src", help = "Path like `/Path/to/ChallengeData`")
    parser.add_argument("--tgt", help = "Path like `/Path/to/h5`")
    parser.add_argument('--year', type=int, required=True, choices=[2024, 2023, 2025], help='year of the dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of worker processes for parallel processing')

    args = parser.parse_args()
    
    mat_folder = args.src
    save_folder = args.tgt
    year = args.year
    workers = args.workers
    
    print('matlab data folder: ', mat_folder)
    print('h5 save folder: ', save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    print('## step 1: convert matlab training dataset to h5 dataset')

    filelist = glob(os.path.join(mat_folder, "**/*.mat"), recursive=True)
    filelist = [str(file) for file in filelist if 'FullSample' in str(file) and 'Mask' not in str(file)]
    random.shuffle(filelist)  # Shuffle the file list for randomness
    
    print('number of total matlab files: ', len(filelist))

    def process_file(ff):
        ##* get info from path
        if year == 2024:
            # "CMRxRecon2024/MultiCoil/Mapping/TestSet/FullSample/P001/xxx.mat"
            match = re.search(r'MultiCoil/([^/]+)/([^/]+)/([^/]+)/([^/]+)', ff)
            modal = match.group(1)
            setname = match.group(2)
            sampletype = match.group(3)
            center = "Center100"
            mridevice = "Unknown"
            paid = match.group(4)
            directory, filename = os.path.split(ff)
            basename = os.path.splitext(filename)[0]
            save_name = f'{modal}@{setname}@{sampletype}@{center}@{mridevice}@{paid}@{basename}'

        elif year == 2025:
            # "BlackBlood/ValidationSet/UnderSample_TaskR1/Center006/Siemens_30T_Prisma/P007/blackblood_kus_ktRadial8.mat"
            match = re.search(r'MultiCoil/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)', ff)
            modal = match.group(1)
            setname = match.group(2)
            sampletype = match.group(3)
            center = match.group(4)
            mridevice = match.group(5)
            paid = match.group(6)
            directory, filename = os.path.split(ff)
            basename = os.path.splitext(filename)[0]
            save_name = f'{modal}@{setname}@{sampletype}@{center}@{mridevice}@{paid}@{basename}'
        else:
            raise ValueError("year must be in [2024, 2025]")

        ##* load kdata
        with h5py.File(ff, 'r') as f:
            kdata = f['kspace_full']
            kdata = kdata['real'] + 1j * kdata['imag']

        ##* swap phase_encoding and readout
        kdata = torch.as_tensor(kdata.swapaxes(-1, -2))
        
        ##* get rss from kdata
        img_coil = ktoi(kdata)
        img_rss = (img_coil.abs() ** 2).sum(dim=-3, keepdim=False).sqrt().numpy()

        ##* save h5
        save_path = os.path.join(save_folder, save_name + '.h5')
        with h5py.File(save_path, 'w') as file:
            file.create_dataset('kspace', data=kdata)
            file.create_dataset('reconstruction_rss', data=img_rss)
            file.attrs['max'] = img_rss.max()
            file.attrs['acquisition'] = modal
            file.attrs['shape'] = kdata.shape
            file.attrs['patient_id'] = paid
            file.attrs['center'] = center
            file.attrs['mridevice'] = mridevice

        return save_name

    results = pqdm(filelist, process_file, n_jobs=workers, desc="Processing Files")

    print("Processed files:", results)