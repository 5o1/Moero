###################
# Splitting the local validation set from fastmri's dataset
#
# multicoil_train -> train, val0
# multicoil_val -> test
###################


import os
import h5py
from glob import glob
import random

###################
# configs
src_train_path = "path-to/FastMRI-Knee/multicoil_train"
src_val_path = "path-to/FastMRI-Knee/multicoil_val"

tgt_train_path = "path-to/fastmrimix/knee-train"
tgt_val0_path = "path-to/fastmrimix/knee-val0"
tgt_test_path = "path-to/fastmrimix/knee-test"

trainset_weight = 9
val0set_weight = 1
####################

# list all subjects
src_train_fnamelist = glob(f"{src_train_path}/*.h5")
src_val_fnamelist = glob(f"{src_val_path}/*.h5")

print(f"Number of subjects: {len(src_train_fnamelist)}")

# split train/val0
random.shuffle(src_train_fnamelist)
train_fnamelist = src_train_fnamelist[: int(len(src_train_fnamelist) * trainset_weight / (trainset_weight + val0set_weight))]
val0_fnamelist = src_train_fnamelist[int(len(src_train_fnamelist) * trainset_weight / (trainset_weight + val0set_weight)) :]

print(f"Number of training subjects: {len(train_fnamelist)}")
print(f"Number of val0 subjects: {len(val0_fnamelist)}")

# make symbolic links
os.makedirs(f"{tgt_train_path}", exist_ok=True)
os.makedirs(f"{tgt_val0_path}", exist_ok=True)
os.makedirs(f"{tgt_test_path}", exist_ok=True)

for fname, tgt_dir_path in zip(
    [train_fnamelist, val0_fnamelist, src_val_fnamelist],
    [tgt_train_path, tgt_val0_path, tgt_test_path],
):
    for f in fname:
        base = os.path.basename(f)
        tgt = os.path.join(tgt_dir_path, base)
        if not os.path.exists(tgt):
            os.symlink(f, tgt)
            print(f"Linking {f} to {tgt}")