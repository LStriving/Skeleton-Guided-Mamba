import json
import os
import argparse
import skvideo.io
import numpy as np
import cv2

import sys
sys.path.append("../")
from libs.utils import VideoKeypointProcessor2

def group_seg_for_one_video(results, threshold=0.5):
    '''
    action_names = ['LaryngealVestibuleClosure', 'UESOpen', 'OralDelivery', 'ThroatTransport', 'HyoidExercise', 'ThroatSwallow', 'SoftPalateLift']
    results: json dict contain multiple swallowing actions or in txt format:
        [
            {
                "score": 0.5499753355979919,
                "segment": [
                    4.398298740386963,
                    5.358728885650635
                ],
                "label": "OralDelivery"
            },
            {
                "score": 0.42056044936180115,
                "segment": [
                    14.646139144897461,
                    15.563594818115234
                ],
                "label": "OralDelivery"
            },
            ...
        ]
    return: grouped segments through time:
    [
        # segment 1
        {
            start: ...,
            end: ...,
        },
        ...
    ] 
    '''
    if not results:
        return []
    
    # 按照起始时间对结果进行排序
    sorted_results = sorted(results, key=lambda x: x['segment'][0])
    
    merged = []
    for res in sorted_results:
        current_start, current_end = res['segment']
        if not merged:
            merged.append([current_start, current_end])
        else:
            last_start, last_end = merged[-1]
            # 检查当前时间段是否与合并后的最后一个时间段重叠或邻近
            if current_start <= (last_end + threshold):
                # 合并时间段
                new_start = min(last_start, current_start)
                new_end = max(last_end, current_end)
                merged[-1] = [new_start, new_end]
            else:
                merged.append([current_start, current_end])
    
    # 转换为所需的输出格式
    grouped_segments = [{'start': seg[0], 'end': seg[1]} for seg in merged]
    return grouped_segments

def main(args):
    with open(args.json, 'r') as f:
        data = json.load(f)
        
    segs = group_seg_for_one_video(data)
    print(f'Total {len(segs)} swallowing event(s)')
    print(segs)
    
    os.makedirs(args.save_path, exist_ok=True)
    # read video
    video_data = skvideo.io.vread(args.video)
    video_name = os.path.basename(args.video).split(args.ext)[0]
    # get fps
    videometadata = skvideo.io.ffprobe(args.video)
    fps = eval(videometadata['video']['@avg_frame_rate'])
    print(fps)
    processor = VideoKeypointProcessor2('./pretrained/heatmap/best_model_trace.pt',
                                        sigma=args.sigma)
    copied_video_data = np.copy(video_data)
    # hex2rgb = 
    # keypoints_results = []
    for i, seg in enumerate(segs):
        # slice video data
        start_frame, end_frame = int(fps * seg['start']), int(fps * seg['end'])
        seg_data = video_data[start_frame: end_frame]
        # infer and get keypoint npy
        keypoints, _, _ = processor.infer_heatmaps(seg_data)
        np.save(os.path.join(args.save_path,f'keypoint-{i}-{video_name}.npy'), processor.keypoints)
        # circle the keypoint on the original frame using green colors
        for index in range(start_frame, end_frame):
            keypoints = processor.keypoints[index-start_frame]
            for point in keypoints:# 243,229,30
                img = cv2.circle(video_data[index], (int(point[0]*video_data.shape[1]), int(point[1]*video_data.shape[2])), 5, (0,255,0), -1)
                # replace the orignal frame
                copied_video_data[index] = img
    # save video
    skvideo.io.vwrite(
        os.path.join(args.save_path, f'keypoint_{video_name}.mp4'),
        copied_video_data,
        inputdict = {'-r':str(fps)},
        outputdict={'-r': str(fps)}
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--save_path", default='outputs/')
    parser.add_argument("--ext", type=str, choices=['.avi','.mp4'], default='.avi')
    parser.add_argument("--keypoint_color", type=str, default="#")
    args = parser.parse_args()
    main(args)