import os
from tqdm import tqdm
import numpy as np

def check_shape(v_dir, h_dir):
    assert os.path.exists(v_dir), f"visual data directory not found: {v_dir}"
    assert os.path.exists(h_dir), f"heatmap data directory not found: {h_dir}"

    v_files = os.listdir(v_dir)
    v_files = [f for f in v_files if f.endswith('.npy')]

    inconsist = []

    for f in tqdm(v_files):
        v_path = os.path.join(v_dir, f)
        if not os.path.exists(os.path.join(h_dir, f)):
            inconsist.append(f)
            print(f"File {f} is not found.")
            continue
        h_path = os.path.join(h_dir, f)
        v_data = np.load(v_path)
        h_data = np.load(h_path)
        if v_data.shape[0] != h_data.shape[0]:
            inconsist.append(f)
            print(f"File {f} is inconsistent.")
    
    print(f"Total {len(inconsist)} files are inconsistent.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_dir", type=str, required=True)
    parser.add_argument("--h_dir", type=str, required=True)
    args = parser.parse_args()
    check_shape(args.v_dir, args.h_dir)
'''
python libs/utils/check_shape.py \
    --v_dir tmp/multi_class \
    --h_dir tmp/heatmap

python libs/utils/check_shape.py \
    --v_dir data/swallow/stage_2/rgb_flow_no_interplote/no_interplote \
    --h_dir data/swallow/stage_2/heatmap_only/no_interplote
'''