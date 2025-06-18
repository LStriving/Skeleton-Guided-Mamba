import os
import skvideo.io
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm
import logging
import multiprocessing

def resize_video(video_path, output_path, size=(256, 256)):
    # Load video
    if '.mp4' in video_path or '.avi' in video_path:
        videodata = skvideo.io.vread(video_path)
    elif '.npy' in video_path:
        videodata = np.load(video_path)
    else:
        raise ValueError('ext not supported')
    if videodata.shape[-1] == 3:
        # Resize video
        resized_videodata = np.zeros((videodata.shape[0], size[0], size[1], 3), dtype=np.uint8)
    else:
        resized_videodata = np.zeros((videodata.shape[0], size[0], size[1]), dtype=np.float32)
    for i, frame in enumerate(videodata):
        resized_videodata[i] = cv2.resize(frame, size)
    
    # Save video
    if '.mp4' in video_path or '.avi' in video_path:
        skvideo.io.vwrite(output_path, resized_videodata)
    elif '.npy' in video_path:
        np.save(output_path, resized_videodata)

def centercrop_resize_video(video_path, output_path, size=(224, 224)):
    if size[0] == 224:
        resized_len = 226
    elif size[0] == 128:
        resized_len = 130
    else:
        raise ValueError(f"Unsupported size {size}.")
    # Load video
    videodata = skvideo.io.vread(video_path)
    # Resize video
    resized_videodata = np.zeros((videodata.shape[0], size[0], size[1], 3), dtype=np.uint8)
    for i, frame in enumerate(videodata):
        h, w = frame.shape[:2]
        if w < resized_len or h < resized_len:
            d = resized_len - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        
        resized_videodata[i] = center_crop(frame, size)
    # Save video
    skvideo.io.vwrite(output_path, resized_videodata)

def center_crop(image, output_size):
    h, w = image.shape[:2]
    new_h, new_w = output_size
    i = int(np.round((h - new_h) / 2.))
    j = int(np.round((w - new_w) / 2.))
    image = image[i:i+new_h, j:j+new_w]
    return image

def get_file_list(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            file_list = json.load(f)
            if 'database' in file_list:
                file_list = file_list['database']
        file_list = [file_list.keys()]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            file_list = f.readlines()
        file_list = [file.strip() for file in file_list]
    return file_list


def resize_videos(input_dir, output_dir, size=(256, 256), file_ext='.avi', filter_file=None, resume=False):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    if filter_file is not None:
        if not os.path.exists(filter_file):
            raise FileNotFoundError(f"Filter file {filter_file} does not exist.")
        else:
            videos = get_file_list(filter_file)
            logging.info(f"Filtering {len(videos)} videos.")
    else:
        videos = [video for video in os.listdir(input_dir) if video.endswith(file_ext)]
    
    # 使用 Manager 来共享进度
    manager = multiprocessing.Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    total_videos = len(videos)

    def update_progress():
        progress.value += 1

    if args.num_workers > 1:
        pool = multiprocessing.Pool(args.num_workers)
        for video_name in videos:
            video_name = str(video_name).strip()
            if not video_name.endswith(file_ext):
                video_name += file_ext
            video_path = os.path.join(input_dir, video_name)
            output_path = os.path.join(output_dir, video_name)
            if resume and os.path.exists(output_path):
                try:
                    if '.npy' in output_path:
                        np.load(output_path)
                    elif '.avi' in output_path:
                        skvideo.io.vread(output_path)
                    logging.info(f"Skipping {video_name}, already processed.")
                    update_progress()  # 更新进度
                    continue
                except Exception as e:
                    ...
            logging.info(f"Resizing {video_name}...")
            pool.apply_async(resize_video, args=(video_path, output_path, size), callback=lambda _: update_progress())
        
        pool.close()
        pool.join()

    # 打印最终进度
    for _ in tqdm(range(total_videos), desc="Processing videos"):
        while progress.value < total_videos:
            pass  # 等待所有进程完成
    logging.info("All videos processed.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--filter_file", type=str, default=None)
    parser.add_argument('--size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--file_ext', type=str, default='.avi')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--center_crop', action='store_true')
    args = parser.parse_args()
    if args.center_crop:
        resize_video = centercrop_resize_video
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")
    resize_videos(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        size=args.size, 
        file_ext=args.file_ext, 
        filter_file=args.filter_file, 
        resume=args.resume)