import os
from tqdm import tqdm
import numpy as np

def concat_feat(v_feat_root, h_feat_root, out_feat_root):
    os.makedirs(out_feat_root, exist_ok=True)
    for v_feat_file in tqdm(os.listdir(v_feat_root)):
        if not v_feat_file.endswith('.npy'):
            continue
        assert v_feat_file in os.listdir(h_feat_root), f"{v_feat_file} not in {h_feat_root}"
        v_feat = np.load(os.path.join(v_feat_root, v_feat_file))
        h_feat = np.load(os.path.join(h_feat_root, v_feat_file))
        if v_feat.shape[0] != h_feat.shape[0]:
            print(f"Shapes of {v_feat_file} do not match: {v_feat.shape} vs {h_feat.shape}")
            continue
        fused_feat = np.concatenate([v_feat, h_feat], axis=1)
        out_feat_file = v_feat_file
        np.save(os.path.join(out_feat_root, out_feat_file), fused_feat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--v_feat_root', type=str, required=True)
    parser.add_argument('--h_feat_root', type=str, required=True)
    parser.add_argument('--out_feat_root', type=str, required=True)
    args = parser.parse_args()
    concat_feat(args.v_feat_root, args.h_feat_root, args.out_feat_root)
'''
python libs/utils/concat_feat.py \
        --v_feat_root tmp/mulmodal_new \
        --h_feat_root tmp/heatmap_orirgbi3d \
        --out_feat_root tmp/3modal


python libs/utils/concat_feat.py \
    --v_feat_root ./feat/rgb_224/no_interplote \
    --h_feat_root data/swallow/stage_1/flow \
    --out_feat_root data/swallow/stage_1/224
'''