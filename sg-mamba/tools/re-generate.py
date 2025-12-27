'''Script to recover Stage 2 video clips from Stage 1 videos based on JSON annotations.
'''
import json
import os
import subprocess
import argparse
import csv
import logging
import sys
import numpy as np

# Paths to your JSON annotation files
STAGE1_JSON_PATH = 'data/swallow/anno/swallow_singlestage.json'
STAGE2_JSON_PATH = 'data/swallow/anno/swallow_stage2_trainval.json'

# Paths to video folders
STAGE1_VIDEO_DIR = 'data/swallow/sliding_videos' 
OUTPUT_DIR = 'data/swallow/stage_2'

# ================= CONFIGURATION =================
# Tolerance for temporal matching (seconds)
TIME_TOLERANCE = 0
FPS = 29.97002997
# =================================================

def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON at {path}: {e}")
        sys.exit(1)

def extract_common_id(filename_key):
    """
    Extracts the shared identifier string from the filename.
    Format S2: 1_13.5..._2021010602_han..._102926_4
    Format S1: 1_0_2021010602_han..._102926_32
    Target: 2021010602_han..._102926
    """
    parts = filename_key.split('_')
    
    # Heuristic: The shared ID is usually everything from index 2 up to the last index.
    # We check length to ensure it fits the expected format.
    if len(parts) > 3:
        # Join from index 2 up to the second to last part (excluding the suffix ID like _4 or _32)
        # However, looking at your example:
        # S2 End: ..._102926_4 -> We want up to 102926.
        # Python slice [2:-1] grabs from index 2 up to (but not including) the last element.
        return "_".join(parts[2:-1])
    
    # Fallback if filename is weird: return the whole thing to force strict matching
    return filename_key

def get_temporal_fingerprint(annotation_list):
    """
    Generates a shift-invariant fingerprint (Duration + Relative Start Time).
    """
    actions = [a for a in annotation_list if a['label'] != 'AllTime']
    actions.sort(key=lambda x: x['segment(frames)'][0])
    
    if not actions:
        return None

    anchor_time = actions[0]['segment(frames)'][0]
    fingerprint = []

    for action in actions:
        fingerprint.append({
            'label': action['label'],
            'duration': action['segment(frames)'][1] - action['segment(frames)'][0],
            'rel_start': action['segment(frames)'][0] - anchor_time,
            'abs_start': action['segment(frames)'][0]
        })
    return fingerprint

def calculate_match_score(fp_s2, fp_s1):
    """
    Compares S2 fingerprint against S1. Returns (True/False, Offset).
    """
    if len(fp_s2) > len(fp_s1): return False, 0.0 # S1 can't have fewer actions than S2

    # We assume S2 is a subset or exact match of S1 sequence.
    # We try to align the first action of S2 with every possible action in S1
    for i in range(len(fp_s1) - len(fp_s2) + 1):
        
        # Check if the sequence matches starting at index i of S1
        offsets = []
        is_seq_match = True
        
        for k in range(len(fp_s2)):
            a2 = fp_s2[k]
            a1 = fp_s1[i + k] # Corresponding action in S1

            # 1. Label Check
            if a2['label'] != a1['label']:
                is_seq_match = False
                break
            
            # 2. Duration Check
            if abs(a2['duration'] - a1['duration']) > TIME_TOLERANCE:
                is_seq_match = False
                break
            
            # 3. Relative Structure Check (Timing between actions)
            # Compare (current S2 rel_start) vs (current S1 rel_start - S1_anchor_offset)
            # Easier way: The difference in start times must be consistent.
            
            start_diff = a1['abs_start'] - a2['abs_start']
            offsets.append(start_diff)
            
            # Consistency check: Compare this offset with the first item's offset
            if k > 0:
                if abs(start_diff - offsets[0]) > TIME_TOLERANCE:
                    is_seq_match = False
                    break
        
        if is_seq_match:
            return True, np.median(offsets) / FPS

    return False, 0.0

def cut_video(source_path, output_path, start_time, duration):
    """
    Uses FFmpeg to cut the video.
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', f"{start_time}",
        '-i', source_path,
        '-t', f"{duration}",
        '-c:v', 'libx264', '-c:a', 'aac', 
        '-strict', 'experimental',
        '-loglevel', 'error',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, str(e)

def main():
    global TIME_TOLERANCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1_json', default=STAGE1_JSON_PATH)
    parser.add_argument('--s2_json', default=STAGE2_JSON_PATH)
    parser.add_argument('--video_dir', default=STAGE1_VIDEO_DIR)
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument('--csv_report', default="recovery_report.csv")
    parser.add_argument('--log_file', default='logs/recovery.log')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--time_tolerance', '-t', type=float, default=TIME_TOLERANCE, help='Time tolerance in seconds for matching')
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    # 1. Load Data
    s1_data = load_json(args.s1_json)
    s2_data = load_json(args.s2_json)
    TIME_TOLERANCE = args.time_tolerance

    # 2. Build S1 Lookup Table (Key: CommonIDString -> List of S1 Items)
    logging.info("Indexing Stage 1 videos by ID string...")
    s1_lookup = {}
    
    for key, info in s1_data.items():
        common_id = extract_common_id(key)
        fp = get_temporal_fingerprint(info['annotations'])
        
        if fp:
            if common_id not in s1_lookup:
                s1_lookup[common_id] = []
            s1_lookup[common_id].append({'key': key, 'fp': fp})
            
    logging.info(f"Indexed {len(s1_data)} S1 videos into {len(s1_lookup)} unique ID groups.")

    # 3. Process
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    processed = 0
    matched_count = 0
    
    print("-" * 60)
    for s2_key, s2_info in s2_data.items():
        processed += 1
        s2_common_id = extract_common_id(s2_key)
        s2_fp = get_temporal_fingerprint(s2_info['annotations'])
        
        status = "Failed"
        match_key = None
        offset = 0.0
        note = ""

        if not s2_fp:
            note = "No micro-actions"
        elif s2_common_id not in s1_lookup:
            note = f"ID '{s2_common_id}' not found in Stage 1"
        else:
            # We have candidate parents!
            candidates = s1_lookup[s2_common_id]
            
            # If only 1 candidate, we still verify with fingerprint to calculate offset
            # If multiple, fingerprint finds the correct one among them.
            best_match = None
            
            for candidate in candidates:
                is_match, calc_offset = calculate_match_score(s2_fp, candidate['fp'])
                if is_match:
                    best_match = candidate
                    offset = max(0.0, calc_offset)
                    match_key = candidate['key']
                    break 
            
            if match_key:
                status = "Matched"
            else:
                note = f"ID matched ({len(candidates)} candidates), but temporal patterns differed."

        # Execute
        if status == "Matched":
            # Find File
            s1_path = os.path.join(args.video_dir, match_key + ".avi")
            if not os.path.exists(s1_path):
                 # Try prefix search
                 cands = [f for f in os.listdir(args.video_dir) if f.startswith(match_key)]
                 if cands: s1_path = os.path.join(args.video_dir, cands[0])
                 else: 
                     status = "Missing Src"
                     note = "File not found on disk"
            
            if status == "Matched" and not args.dry_run:
                success, msg = cut_video(s1_path, os.path.join(args.output_dir, s2_key + ".avi"), offset, s2_info['duration'])
                status = "Success" if success else "FFmpeg Error"
                if not success: note = msg
                else: logging.info(f"[OK] {s2_key} -> {match_key} (Offset: {offset:.3f}s)")
            elif args.dry_run:
                logging.info(f"[DRY] {s2_key} matches {match_key} (Offset: {offset:.3f}s)")

        if status in ["Success", "Matched"]: matched_count += 1
        else: 
            if args.verbose: logging.warning(f"Fail {s2_key}: {note}")

        results.append({
            'Stage2_ID': s2_key,
            'Matched_S1': match_key,
            'Common_ID_Str': s2_common_id,
            'Offset': round(offset, 4),
            'Status': status,
            'Note': note
        })

    # Save Report
    with open(args.csv_report, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        w.writeheader()
        w.writerows(results)
        
    logging.info(f"\nDone. {matched_count}/{processed} recovered. Report: {args.csv_report}")

if __name__ == "__main__":
    main()