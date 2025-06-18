import os
import numpy as np
import argparse
from tqdm import tqdm

def split_feat(feat_root, out_feat_root, num_splits=2, save_split_idx=1, split_dim=0, transpose=False):
    os.makedirs(out_feat_root, exist_ok=True)
    for feat_file in tqdm(os.listdir(feat_root)):
        if not feat_file.endswith('.npy'):
            continue
        feat = np.load(os.path.join(feat_root, feat_file)) # T, C
        if transpose:
            feat = feat.transpose(1, 0) # C, T
        feat_splits = np.array_split(feat, num_splits, axis=split_dim)[save_split_idx]
        out_feat_file = feat_file
        np.save(os.path.join(out_feat_root, out_feat_file), feat_splits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_root', type=str, required=True)
    parser.add_argument('--out_feat_root', type=str, required=True)
    parser.add_argument('--num_splits', type=int, default=2)
    parser.add_argument('--save_split_idx', type=int, default=1)
    parser.add_argument('--split_dim', type=int, default=0)
    parser.add_argument('--transpose', action='store_true')
    args = parser.parse_args()
    split_feat(args.feat_root, args.out_feat_root, args.num_splits, args.save_split_idx, args.split_dim, args.transpose)

"""
Usage:
python tools/split_flow.py \
    --feat_root data/swallow/stage_1 \
    --out_feat_root  data/swallow/stage_1/flow \
    --num_splits 2 \
    --save_split_idx 1 \
    --split_dim 1 \
    # --transpose
"""