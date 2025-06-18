import os
from tqdm import tqdm

def rename_video(video_path, new_name):
    os.rename(video_path, new_name)

root='./data/swallow/stage_2/raw_heatmap_sigma4_line'

for video_name in tqdm(os.listdir(root)):
    video_path = os.path.join(root, video_name)
    new_name = video_path.replace('.avi', '')
    rename_video(video_path, new_name)