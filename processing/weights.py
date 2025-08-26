import torch
import os
import sys

def extract_and_save_state_dict(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint is not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Checkpoint keys: ", checkpoint.keys())
    
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'state_dict': {checkpoint_path}")
    
    new_checkpoint = {
        "state_dict": checkpoint["state_dict"],
        "pytorch-lightning_version": checkpoint["pytorch-lightning_version"]
        }
    
    new_file_name = f"{checkpoint_path}.weights"
    
    if os.path.exists(new_file_name):
        raise FileExistsError(f"The target file already exists: {new_file_name}")

    torch.save(new_checkpoint, new_file_name)
    print(f"state_dict is saved to: {new_file_name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python extract_state_dict.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]

    extract_and_save_state_dict(checkpoint_path)