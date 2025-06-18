import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def load_annotations(json_path):
    """加载JSON标注文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'database' in data:
        data = data['database']
    return data

def is_action_valid(action, window_start, window_end):
    """验证单个动作是否满足条件"""
    # 条件a: 动作完整包含在窗口中
    if action['segment'][0] < window_start or action['segment'][1] > window_end:
        return False
    
    # 条件b: 持续时间大于5帧
    duration_frames = action['segment(frames)'][1] - action['segment(frames)'][0]
    return duration_frames >= 5

def process_window(video_path, window_start, window_duration, output_dir, clip_id, annotations, fps, dry_run=False, to_mp4=False):
    """处理单个时间窗口"""
    window_end = window_start + window_duration
    valid_actions = []
    
    # 筛选有效动作
    for action in annotations['annotations']:
        if is_action_valid(action, window_start, window_end):
            # 创建新的动作记录
            new_action = {
                "label": action["label"],
                "segment": [
                    action["segment"][0] - window_start,
                    action["segment"][1] - window_start
                ],
                "segment(frames)": [
                    max(0, round((action["segment"][0] - window_start) * fps)),
                    round((action["segment"][1] - window_start) * fps)
                ],
                "label_id": action["label_id"]
            }
            valid_actions.append(new_action)
    
    # 包含完整的吞咽动作
    if len(valid_actions) % 7 != 0:
        return None
    
    # 如果没有有效动作则跳过
    if not valid_actions:
        return None
    
    # 生成输出文件名
    video_name = Path(video_path).stem
    output_name = f"{video_name}_clip{clip_id}_{window_duration}s"
    if to_mp4:
        output_video = Path(output_dir) / "videos" / f"{output_name}.mp4"
    else:
        output_video = Path(output_dir) / "videos" / f"{output_name}.avi"
        
    
    # 创建输出目录
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用ffmpeg进行画面裁剪（明心康复中心医院设备）
    crop_params = "crop=613:613:422:211"  # 根据实际需求修改
    
    if to_mp4:
        cmd = [
        'ffmpeg',
        '-ss', str(window_start),
        '-i', video_path,
        '-to', str(window_duration),
        '-vf', crop_params,
        '-c:v', 'libx264',       # 指定视频编码器
        '-preset', 'medium',     # 平衡编码速度和质量
        '-crf', '23',            # 控制视频质量（0-51，值越小质量越高）
        '-pix_fmt', 'yuv420p',   # 确保兼容性
        '-an',                   # 禁用音频
        '-loglevel', 'error',
        '-y', str(output_video)
        ]
    else:
        cmd = [
        'ffmpeg',
        '-ss', str(window_start),
        '-i', video_path,
        '-to', str(window_duration),
        '-vf', crop_params,
        '-loglevel', 'error',
        '-y', str(output_video)
        ]
    if not dry_run:
        subprocess.run(cmd, check=True)
    
    return {
        "video_path": str(output_video.relative_to(output_dir)),
        "duration": window_duration,
        "fps": fps,
        "original_window": [window_start, window_end],
        "annotations": valid_actions
    }

def process_video(video_path, json_data, output_dir, window_sizes, dry_run=False, to_mp4=False):
    """处理单个视频"""
    video_name = Path(video_path).stem
    video_info = json_data[video_name]
    fps = video_info['fps']
    total_duration = video_info['duration']
    
    all_clips = []
    clip_id = 0
    invalid = 0
    
    # 遍历所有窗口尺寸
    for window_size in window_sizes:
        # 计算滑动步长（窗口大小的1/4）
        step = window_size / 4
        # 计算可能的窗口数量（包含结尾不完整窗口）
        num_windows = int((total_duration - window_size) // step) + 1
        
        # 处理每个窗口
        for i in tqdm(range(num_windows), desc=f"Processing {window_size}s windows"):
            window_start = round(i * step, 2)  # 保留两位小数避免精度问题
            window_start = min(window_start, total_duration - window_size)  # 边界保护
            
            clip_info = process_window(
                video_path, window_start, window_size,
                output_dir, clip_id, video_info, fps, dry_run=dry_run, to_mp4=to_mp4
            )
            
            if clip_info:
                all_clips.append(clip_info)
                clip_id += 1
            else:
                invalid += 1
    print(f'video_path: {clip_id+1} valid, {invalid} invalid')
    
    return all_clips

def main():
    parser = argparse.ArgumentParser(description='数据集滑窗处理')
    parser.add_argument('--video_dir', default='data/swallow/external_videos', help='原始视频目录')
    parser.add_argument('--json_path', default='data/swallow/external_videos/valid_videos.json', help='标注文件路径')
    parser.add_argument('--output_dir', default='data/swallow/external_processed', help='输出目录')
    parser.add_argument('--window_sizes', nargs='+', type=int, default=[32, 64],
                        help='滑窗尺寸列表，默认[32, 64]')
    parser.add_argument('--dry_run', action='store_true', help='dry run模式,不进行视频操作,只输出json')
    parser.add_argument('--to_mp4', action='store_true')
    args = parser.parse_args()
    
    # 加载标注数据
    json_data = load_annotations(args.json_path)
    
    # 创建主输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理所有视频
    all_results = {}
    for video_file in Path(args.video_dir).glob('*.avi'):
        video_name = video_file.stem
        if video_name in json_data:
            print(f"\nProcessing video: {video_name}")
            clips = process_video(
                str(video_file),
                json_data,
                args.output_dir,
                args.window_sizes,
                dry_run=args.dry_run,
                to_mp4=args.to_mp4,
            )
            all_results[video_name] = clips
    
    # 保存最终标注文件
    output_json = Path(args.output_dir) / "annotations.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n处理完成！结果已保存至：{args.output_dir}")

if __name__ == "__main__":
    main()