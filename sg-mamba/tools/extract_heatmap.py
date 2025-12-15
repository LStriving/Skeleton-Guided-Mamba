import os
from tqdm import tqdm
import numpy as np
import argparse
import sys
sys.path.append('.')
from libs.utils import VideoKeypointProcessor
from libs.utils import VideoKeypointProcessor2
import cv2
from matplotlib import pyplot as plt

def get_file_list(file_path, ext='.avi'):
    with open(file_path, 'r') as f:
        files = f.readlines()
    files = [file.strip().split(",")[0] for file in files]
    files = [file + '.avi' for file in files if not file.endswith(ext)]
    return files

def main(args):
    sigma = args.sigma
    if args.img_width is not None and args.img_height is not None:
        processor = VideoKeypointProcessor('./ckpts/best_model_trace.pt',
                                        image_height=args.img_height, image_width=args.img_width,
                                        sigma=sigma)
    else:
        processor = VideoKeypointProcessor('./ckpts/best_model_trace.pt',
                                        sigma=sigma)
    input_dir = args.input_dir
    output_dir = args.output_dir
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist."
    assert input_dir != output_dir, "Input and output directories cannot be the same."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.filter_file is None:
        videos = [video for video in os.listdir(input_dir) if video.endswith(args.video_ext)]
    else:
        videos = get_file_list(args.filter_file, args.video_ext)
        print(f"Filtering {len(videos)} videos.")
    for video_name in tqdm(videos):
        video_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name)
        output_path = output_path.replace(args.video_ext, '')
        if os.path.exists(output_path) and not args.overwrite:
            continue
        if args.feature_type == 'keypoint':
            keypoint, _, _ = processor.infer_heatmaps(video_path)
            np.save(output_path, keypoint)
        elif args.feature_type == 'line':
            _, line, _ = processor.infer_heatmaps(video_path)
            np.save(output_path, line)
        elif args.feature_type == 'fusion':
            _, _, cropped_fusion = processor.infer_heatmaps(video_path)
            np.save(output_path, cropped_fusion)
        elif args.feature_type == 'all':
            keypoint, line, cropped_fusion = processor.infer_heatmaps(video_path)
            np.save(output_path + '_keypoint', keypoint)
            np.save(output_path + '_line', line)
            np.save(output_path + '_fusion', cropped_fusion)
        else:
            raise ValueError(f"Unknown feature type: {args.feature_type}")

    print(f"Processed {len(videos)} videos.")
    return

def infer_single_image(args):
    if args.no_crop:
        processor = VideoKeypointProcessor2('./ckpts/best_model_trace.pt',
                                        sigma=args.sigma, crop_mode='none')
    else:
        processor = VideoKeypointProcessor2('./ckpts/best_model_trace.pt',
                                        sigma=args.sigma)
    input_image_path = args.image_path
    output_image_path = args.output_image_path
    assert os.path.exists(input_image_path), f"Input image {input_image_path} does not exist."
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    input_img_data = cv2.imread(input_image_path)
    # to RGB
    input_img_data = cv2.cvtColor(input_img_data, cv2.COLOR_BGR2RGB)
    # to numpy array
    input_img_data = np.array(input_img_data)
    # unsqueeze to add batch dimension
    input_img_data = np.expand_dims(input_img_data, axis=0)
    # infer heatmaps
    selected_index = None
    if args.feature_type == 'keypoint':
        selected_index = 0
    elif args.feature_type == 'line':
        selected_index = 1
    elif args.feature_type == 'fusion':
        selected_index = 2
    elif args.feature_type == 'all':
        selected_index = None
    else:
        raise ValueError(f"Unknown feature type: {args.feature_type}")
    
    keypoint, line, cropped_fusion = processor.infer_heatmaps(input_img_data)
    # save heatmaps
    if selected_index is not None:
        heatmaps = [keypoint, line, cropped_fusion][selected_index]
        np.save(output_image_path.replace('.png', '.npy'), heatmaps)
        np.save(output_image_path.replace('.png', '_keypoints.npy'),processor.keypoints)
        plt.imshow(heatmaps[0])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Keypoint Heatmap')
        plt.imshow(keypoint[0])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Line Heatmap')
        plt.imshow(line[0])
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Fusion Heatmap')
        plt.imshow(cropped_fusion[0])
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    print(f"Heatmap saved to {output_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument("--filter_file", type=str, default=None)
    parser.add_argument('--sigma', type=float, default=4)
    parser.add_argument("--video_ext", type=str, default='.avi')
    parser.add_argument("--feature_type", choices=['keypoint', 'line', 'fusion','all'], default='fusion')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument('--img_input', action='store_true',
                        help='Whether to process a single image instead of a video.')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the input image file.')
    parser.add_argument('--output_image_path', type=str, default=None,
                        help='Path to save the output heatmap image.')
    parser.add_argument('--no_crop', action='store_true',
                        help='Whether to disable cropping of the input image.')
    args = parser.parse_args()
    if args.img_input:
        infer_single_image(args)
    else:
        main(args)

'''
python tools/extract_heatmap.py \
    --input_dir datas \
    --output_dir tmp/plot \
    --filter_file tmp/tmplist \
    --sigma 4 \
    --feature_type all \
    --img_width 612 \
    --img_height 612

# single image inference
python tools/extract_heatmap.py \
    --img_input \
    --image_path assets/image.png \
    --output_image_path tmp/plot/single_image_heatmap.png \
    --sigma 4 \
    --feature_type fusion \
    --img_width 612 \
    --img_height 612 \
    --no_crop
'''