import json
import tqdm
import skvideo.io
import numpy as np
import os

def crop_video(video_path, output_path, start_time: float, end_time: float):
    video = skvideo.io.vread(video_path)
    fps = 29.97002997002997
    frame_cnt = len(video)

    start_frame = int(start_time * float(fps))
    start_frame = max(start_frame, 0)
    assert start_frame < frame_cnt
    end_frame = int(end_time * float(fps))
    end_frame = min(end_frame, frame_cnt)

    cropped_video = video[start_frame:end_frame]
    os.path.makedirs(os.path.dirname(output_path), exist_ok=True)
    skvideo.io.vwrite(output_path, cropped_video)


def crop_videos(video_dir, output_dir, time_file, resume=False):
    with open(time_file, 'r') as f:
        time_dict = json.load(f)

    assert os.path.exists(video_dir)
    os.makedirs(output_dir, exist_ok=True)

    for seg_id, anno in tqdm.tqdm(time_dict.items()):
        video_name = seg_id.split("#")[0] + '.avi'
        seg_name = seg_id + '.avi'
        video_path = os.path.join(video_dir, video_name)
        output_path = os.path.join(output_dir, seg_name)
        if os.path.exists(output_path) and resume:
            continue

        start_time, end_time = anno['shift'], anno['shift'] + 4
        try:
            crop_video(video_path, output_path, start_time, end_time)
        except Exception as e:
            print(f"Error in {video_path}:\n start_time: {start_time}, end_time: {end_time}\n{e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crop videos')
    parser.add_argument('--video_dir', type=str, help='path to video directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--time_file', type=str, help='path to time file')
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    args = parser.parse_args()

    crop_videos(args.video_dir, args.output_dir, args.time_file, args.resume)

'''
python tools/crop_videos.py \
    --video_dir datas \
    --output_dir tmp/stage1_val82_threshold0.3_avi/ \
    --time_file ./tmp/multi_class/tmp.json \
    --resumes
'''