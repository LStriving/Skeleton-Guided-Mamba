import os
import argparse
import json
import cv2
from tqdm import tqdm

def check_video(video_path, expected_duration, fps):
    """
    Check if a video file is valid and its duration matches the annotation.
    :param video_path: Path to the video file.
    :param expected_duration: Expected duration of the video in seconds.
    :param fps: Frames per second of the video.
    :return: Tuple (is_valid, reason). is_valid is True if the video is valid, False otherwise.
             reason is a string explaining why the video is invalid.
    """
    if not os.path.exists(video_path):
        return False, "Video file not found"

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_duration = frame_count / actual_fps if actual_fps > 0 else 0

        if abs(actual_duration - expected_duration) > 1:  # Allow a small margin of error
            return False, f"Duration mismatch (expected: {expected_duration}, actual: {actual_duration:.2f})"

        cap.release()
        return True, "Valid"
    except Exception as e:
        return False, f"Error reading video: {str(e)}"

def check_dataset(dataset_path, anno_file):
    """
    Check the dataset for missing or broken videos.
    :param dataset_path: Path to the dataset directory.
    :param anno_file: Path to the annotation JSON file.
    """
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    lost_videos = []
    broken_videos = []

    for video_id, video_info in tqdm(annotations.items(), desc="Checking videos"):
        subset = video_info.get("subset", "Unknown")
        duration = video_info.get("duration", 0)
        fps = video_info.get("fps", 0)
        video_file = os.path.join(dataset_path, f"{video_id}.avi")

        is_valid, reason = check_video(video_file, duration, fps)
        if not is_valid:
            if reason == "Video file not found":
                lost_videos.append((video_id, subset))
            else:
                broken_videos.append((video_id, subset, reason))

    # Generate report
    print("\n=== Dataset Check Report ===")
    print(f"Total videos in annotation: {len(annotations)}")
    print(f"Lost videos: {len(lost_videos)}")
    for video_id, subset in lost_videos:
        print(f"  - {video_id} (subset: {subset})")

    print(f"Broken videos: {len(broken_videos)}")
    for video_id, subset, reason in broken_videos:
        print(f"  - {video_id} (subset: {subset}, reason: {reason})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset for missing or broken videos.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--anno", type=str, required=True, help="Path to the annotation JSON file")
    args = parser.parse_args()

    check_dataset(args.dataset, args.anno)