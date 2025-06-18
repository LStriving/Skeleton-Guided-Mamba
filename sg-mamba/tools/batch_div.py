import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def div_feature(feature_path, save_path):
    feature = np.load(feature_path)
    feature = feature / 2
    np.save(save_path, feature)

def worker(feature_name):
    feature_path = os.path.join(root, feature_name)
    save_path = os.path.join(save_root, feature_name)
    if os.path.exists(save_path):
        return
    div_feature(feature_path, save_path)

if __name__ == '__main__':
    root = 'tmp/raw_heatmap_sigma4_line'
    save_root = 'tmp/raw_heatmap_sigma4_line_div2'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    feature_names = os.listdir(root)
    feature_names = [f for f in feature_names if f.endswith('.npy')]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(worker, feature_names), total=len(feature_names)))
