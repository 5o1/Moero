import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a text file with features to a compressed .npz file.  Usage: python txttonp.py <src_txt_file> <tgt_npz_file>"
    )
    parser.add_argument(
        'src',
        type=str,
        help='Path to the source text file containing features.'
    )
    parser.add_argument(
        'tgt',
        type=str,
        help='Path to the target .npz file to save the features.'
    )
    args = parser.parse_args()
    src_path = args.src
    tgt_path = args.tgt


    # Read lines from the text file
    with open(src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # fname islices freatures
    lines_split = [line.strip().split() for line in lines]
    fnames = [line[0] for line in lines_split]
    islices = [int(line[1]) for line in lines_split]
    features = [list(map(float, line[2:])) for line in lines_split]

    # Save to .np file
    np.savez_compressed(tgt_path, fnames=fnames, islices=islices, features=features)