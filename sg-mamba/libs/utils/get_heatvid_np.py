import os
import numpy as np
from tqdm import tqdm
import pickle as pkl
import skvideo.io
import torch
import cv2

def load_res(path):
    assert os.path.exists(path), f'{path} not exists!'
    with open(path,'rb')as f:
        data = pkl.load(f)
    return data['result']

def main(args):
    result = load_res(args.pickle_path)
    CLIP_DUR = 4
    IMAGE_SIZE = 128
    for idx in tqdm(range(len(result['video-id']))):
        video_id = result['video-id'][idx]
        t_center = result['t-center'][idx]
        t_extend = CLIP_DUR # 4 seconds
        clip_start = t_center - t_extend / 2
        clip_end = t_center + t_extend / 2
        duration = int(video_id.split("_")[-1])
        clip_start = max(clip_start, 0)
        clip_end = min(clip_end, duration)
        start_ratio = clip_start / duration
        end_ratio = clip_end / duration
        
        
        if cache_video_id != video_id:
            cache_video_id = video_id
            per_video_rank = 0
            video_path = os.path.join(args.video_root, f"{video_id}.avi")
            assert os.path.isfile(video_path), "Video file does not exist!"
            try:
                rgb_data = skvideo.io.vread(video_path)
            except Exception as e:
                print(f"Error reading video: {video_path}")
                raise e
            # resize video
            rgb_data=torch.from_numpy(rgb_data)
            rgb_data_tmp=torch.zeros(rgb_data.shape[0:1]+(IMAGE_SIZE,IMAGE_SIZE,3)).double()
            for index,rgb_data in enumerate(rgb_data):
                rgb_datum_tmp=torch.from_numpy(cv2.resize(rgb_data.numpy(),(IMAGE_SIZE,IMAGE_SIZE))).double()
                rgb_data_tmp[index,:,:,:]=rgb_datum_tmp
            rgb_data=rgb_data_tmp.view(-1,IMAGE_SIZE,IMAGE_SIZE,3) / 127.5 - 1
        else:
            per_video_rank += 1
    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_feat_root", type=str, default='./tmp/multi_class')
    parser.add_argument('--pickle_path',type=str,default='./tmp/multi_class/epoch_024_0.82621.pkl')
    parser.add_argument('--video_root', type=str,default='datas' )