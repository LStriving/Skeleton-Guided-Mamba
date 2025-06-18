import os
import numpy as np
from tqdm import tqdm
import argparse

def compare_feat(path1, path2):
    feat1 = np.load(path1)
    feat2 = np.load(path2)
    return np.allclose(feat1[:,1024:],feat2[:,1024:],atol=0)
    

def compares_flow_feats(input_dir1, input_dir2):
    assert os.path.exists(input_dir1), f"Input directory {input_dir1} does not exist."
    assert os.path.exists(input_dir2), f"Input directory {input_dir2} does not exist."
    
    feats1 = os.listdir(input_dir1)
    feats2 = os.listdir(input_dir2)

    feats1 = [feat for feat in feats1 if feat.endswith('.npy')]
    feats2 = [feat for feat in feats2 if feat.endswith('.npy')]

    feats1.sort()

    for feat in tqdm(feats1):
        feat1 = os.path.join(input_dir1, feat)
        feat2 = os.path.join(input_dir2, feat)
        if not compare_feat(feat1, feat2):
            print(feat)
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare flow features.')
    parser.add_argument('--input_dir1', type=str, required=True, help='Path to the input directory 1.')
    parser.add_argument('--input_dir2', type=str, required=True, help='Path to the input directory 2.')
    args = parser.parse_args()
    
    print(compares_flow_feats(args.input_dir1, args.input_dir2))