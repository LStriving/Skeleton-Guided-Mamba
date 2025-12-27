import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from pykalman import KalmanFilter
import skvideo.io
import torch.nn.functional as F


def kalman_filter_without_confidence(video_keypoints):
    """
    对视频关键点进行卡尔曼滤波（不考虑置信度）。
    
    参数:
    video_keypoints: np.ndarray，形状为 [n, 8, 2]，表示 n 帧视频中每帧的 8 个关键点 (x, y) 坐标。
    
    返回:
    smoothed_keypoints: np.ndarray，形状为 [n, 8, 2]，滤波后的关键点。
    """
    n_frames, n_keypoints, _ = video_keypoints.shape
    observations = video_keypoints.reshape(n_frames, n_keypoints * 2)

    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=np.eye(n_keypoints * 4),
        observation_matrices=np.hstack([
            np.eye(n_keypoints * 2),
            np.zeros((n_keypoints * 2, n_keypoints * 2))
        ]),
        transition_covariance=0.01 * np.eye(n_keypoints * 4),
        observation_covariance=0.05 * np.eye(n_keypoints * 2),
        initial_state_mean=np.zeros(n_keypoints * 4),
        initial_state_covariance=0.1 * np.eye(n_keypoints * 4)
    )

    # 初始状态 (位置来自第一帧，速度为零)
    initial_state = np.hstack([observations[0], np.zeros(n_keypoints * 2)])
    kf.initial_state_mean = initial_state

    # 应用卡尔曼滤波
    smoothed_states, _ = kf.filter(observations)
    smoothed_keypoints = smoothed_states[:, :n_keypoints * 2].reshape(n_frames, n_keypoints, 2)

    return smoothed_keypoints


def kalman_filter_with_confidence(video_keypoints, confidences):
    """
    对视频关键点进行卡尔曼滤波（考虑置信度）。
    
    参数:
    video_keypoints: np.ndarray，形状为 [n, 8, 2]，表示 n 帧视频中每帧的 8 个关键点 (x, y) 坐标。
    confidences: np.ndarray，形状为 [n, 8]，表示 n 帧中每帧 8 个关键点的置信度。
    
    返回:
    smoothed_keypoints: np.ndarray，形状为 [n, 8, 2]，滤波后的关键点。
    """
    n_frames, n_keypoints, _ = video_keypoints.shape
    observations = video_keypoints.reshape(n_frames, n_keypoints * 2)

    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=np.eye(n_keypoints * 4),
        observation_matrices=np.hstack([
            np.eye(n_keypoints * 2),
            np.zeros((n_keypoints * 2, n_keypoints * 2))
        ]),
        transition_covariance=0.01 * np.eye(n_keypoints * 4),
        initial_state_mean=np.zeros(n_keypoints * 4),
        initial_state_covariance=0.1 * np.eye(n_keypoints * 4)
    )

    # 初始状态 (位置来自第一帧，速度为零)
    initial_state = np.hstack([observations[0], np.zeros(n_keypoints * 2)])
    current_state_mean = initial_state
    current_state_covariance = kf.initial_state_covariance

    max_noise = 1  # 置信度为 0 时的观测噪声
    min_noise = 0  # 置信度为 1 时的观测噪声

    # 动态调整观测噪声
    smoothed_states = []
    for t in range(n_frames):
        observation_noise = np.zeros((n_keypoints * 2, n_keypoints * 2))
        for i in range(n_keypoints):
            confidence = np.clip(confidences[t, i], 0, 1)  # 将置信度限制在 [0, 1]
            # noise = max_noise * (1 - confidence) + min_noise * confidence
            noise = max_noise / (1 + np.exp(20 * (confidence - 0.5))) + min_noise
            observation_noise[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.eye(2) * noise

        # 用当前帧数据进行滤波
        current_state_mean, current_state_covariance = kf.filter_update(
            current_state_mean,
            current_state_covariance,
            observation=observations[t],
            observation_covariance=observation_noise
        )
        smoothed_states.append(current_state_mean)

    smoothed_states = np.array(smoothed_states)
    smoothed_keypoints = smoothed_states[:, :n_keypoints * 2].reshape(n_frames, n_keypoints, 2)

    return smoothed_keypoints

def mixed_keypoints_weighted(video_keypoints, confidences, kalman_smoothed_keypoints, threshold=0.7):
    """
    根据置信度动态加权选择关键点：
    - 置信度 > threshold 时，使用加权融合：weight * video_keypoints + (1 - weight) * kalman_smoothed_keypoints；
    - 置信度 <= threshold 时，使用卡尔曼滤波后的点。

    参数:
    video_keypoints: np.ndarray，形状为 [n, 8, 2]，原始预测的关键点。
    confidences: np.ndarray，形状为 [n, 8]，每个关键点的置信度。
    kalman_smoothed_keypoints: np.ndarray，形状为 [n, 8, 2]，卡尔曼滤波后的关键点。
    threshold: float，置信度阈值，默认值为 0.7。

    返回:
    mixed_keypoints: np.ndarray，形状为 [n, 8, 2]，结合后的关键点。
    """
    # 创建一个布尔掩码，表示置信度 > 阈值的位置
    mask = confidences > threshold  # 形状为 [n, 8]

    # 计算权重，置信度越大，权重越大
    weight = np.clip((confidences - threshold)/(1 - threshold), 0, 1)  # 权重在 [0, 1] 之间

    # 计算加权融合结果
    weighted_keypoints = weight[..., np.newaxis] * video_keypoints + (1 - weight[..., np.newaxis]) * kalman_smoothed_keypoints

    # 选择加权融合结果或者卡尔曼滤波后的点（根据置信度）
    mixed_keypoints = np.where(mask[..., np.newaxis], weighted_keypoints, kalman_smoothed_keypoints)

    return mixed_keypoints


def generate_a_heatmap(arr, centers, max_values, sigma=0.6):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        img_h, img_w = arr.shape
        # max_values = np.ones((len(centers),1))


        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * min(1, max_value)
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

def generate_a_limb_heatmap(arr, starts, ends, start_values, end_values, sigma = 0.6):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        img_h, img_w = arr.shape
        # start_values, end_values = np.ones((len(starts),)), np.ones((len(ends),))
        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                generate_a_heatmap(arr, start[None], start_value[None], sigma)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * min(1, value_coeff)

            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


class VideoKeypointProcessor:
    def __init__(self, model_path, image_width=192, image_height=256, batch_size=32, num_workers=4, sigma=0.6, crop_mode='auto'):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sigma = sigma
        # Load model
        self.model = torch.jit.load(model_path).cuda()
        self.model.eval()

        # Define preprocessing transforms
        self.normalize = transforms.Normalize(
            mean=[0.548, 0.553, 0.551], std=[0.307, 0.307, 0.307]
        )
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        # Attributes to store results
        self.keypoints = None
        self.confidences = None
        self.original_frames = None
        self.skeleton = np.array([(0,1),(0,2),(1,2),(0,4),(3,4),(3,6),(5,6),(5,7),(6,7)])
        self.crop_mode = crop_mode


    def _resize_video_frames(self, video_array):
        """
        Resize all frames of a video to the specified width and height.
        Also returns original frames for visualization.
        """
        videodata = video_array

        resized_frames = []
        original_frames = []

        # 遍历每一帧并调整大小
        for frame in tqdm(videodata, desc="Resizing frames"):
            frame_np = frame  # skvideo 返回的已经是 numpy 数组
            original_frames.append(frame_np)  # 保留原始帧用于可视化
            resized_frame = cv2.resize(frame_np, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)

        return resized_frames, original_frames

    def infer_keypoints(self, video_array, kalman=False, normal_kalman=False):
        """
        Perform inference on the video and store keypoints and confidences.
        """
        resized_frames, self.original_frames = self._resize_video_frames(video_array)
        dataset = [self.preprocess(frame) for frame in resized_frames]
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        outputs_list = []
        for batch in tqdm(data_loader, desc="Inferencing keypoints"):
            batch = batch.cuda()
            with torch.no_grad():
                outputs = self.model(batch)
            outputs_list.append(outputs.cpu().numpy())

        final_output = np.concatenate(outputs_list)
        num_frames, num_joints, height, width = final_output.shape

        # Calculate keypoints and confidences
        indices = np.argmax(final_output.reshape(num_frames, num_joints, -1), axis=-1)
        confidences = np.max(final_output.reshape(num_frames, num_joints, -1), axis=-1)  # Max values as confidence
        y, x = np.unravel_index(indices, (height, width))  # Convert linear indices to (y, x)

        keypoints = np.stack((x, y), axis=-1).astype(np.float32)
        keypoints[:, :, 0] /= width  # Normalize x
        keypoints[:, :, 1] /= height  # Normalize y

        # Store results
        self.keypoints = keypoints
        self.confidences = confidences

        h, w, _ = self.original_frames[0].shape
        if kalman and len(self.keypoints)>1 and not normal_kalman:
            smoothed_keypoints = kalman_filter_with_confidence(self.keypoints, self.confidences)
            reversed_confidences = self.confidences[::-1,:]
            re_smoothed_keypoints = kalman_filter_with_confidence(smoothed_keypoints[::-1,:,:], reversed_confidences)
            # smoothed_keypoints = (smoothed_keypoints + re_smoothed_keypoints[::-1,:,:])*0.5
            smoothed_keypoints = re_smoothed_keypoints[::-1, :, :]
            final_keypoints = mixed_keypoints_weighted(self.keypoints, self.confidences, smoothed_keypoints)
            final_keypoints = kalman_filter_without_confidence(final_keypoints)
            return final_keypoints, confidences, h, w
        elif kalman and len(self.keypoints)>1 and normal_kalman:
            # forward kalman without confidence and only forward
            smoothed_keypoints = kalman_filter_without_confidence(self.keypoints)
            return smoothed_keypoints, confidences, h, w
        elif kalman:
            return self.keypoints, confidences, h, w

        return self.keypoints, self.confidences, h, w
    
    def infer_heatmaps(self, video_array, kalman=True, normal_kalman=False):
        keypoints, confidences, h, w = self.infer_keypoints(video_array, kalman=kalman, normal_kalman=normal_kalman)
        keypoints[:, :, 0] *= w
        keypoints[:, :, 1] *= h  
        cnt = keypoints.shape[0]
        arrs_keypoint = np.zeros((cnt, h, w), dtype=np.float32)
        arrs_edge = np.zeros((cnt, h, w), dtype=np.float32)
        for i in range(cnt):
            arr_keypoint = arrs_keypoint[i, :, :]
            keypoint = keypoints[i]
            arr_edge = arrs_edge[i, :, :]
            starts = keypoint[self.skeleton[:,0],:]
            ends = keypoint[self.skeleton[:,1],:]
            confidence = confidences[i]
            start_values = confidence[self.skeleton[:,0]]
            end_values = confidence[self.skeleton[:,1]]
            generate_a_heatmap(arr_keypoint,keypoint,confidence,sigma=self.sigma)
            generate_a_limb_heatmap(arr_edge,starts,ends,start_values,end_values,sigma=self.sigma)
        
        # 使用 np.nonzero 找到非零元素的索引
        non_zero_indices = np.nonzero(arrs_keypoint)

        if len(non_zero_indices[0]) > 0 and self.crop_mode == 'auto':  # 判断是否存在非零元素
            # 获取最小和最大行列坐标
            min_row, max_row = non_zero_indices[1].min(), non_zero_indices[1].max()  # 行索引
            min_col, max_col = non_zero_indices[2].min(), non_zero_indices[2].max()  # 列索引

            # 基于最小和最大边界裁剪整个批次
            cropped_keypoint = arrs_keypoint[:, min_row:max_row + 1, min_col:max_col + 1]
            cropped_edge = arrs_edge[:, min_row:max_row + 1, min_col:max_col + 1]
        else:
            # 如果整个批次的图像都没有非零元素，可以选择返回全零的图像，或者原图像
            cropped_keypoint = arrs_keypoint
            cropped_edge = arrs_edge
        
        cropped_fusion = (cropped_keypoint+cropped_edge) * 0.5
        return cropped_keypoint, cropped_edge, cropped_fusion

    def visualize_keypoints(self, output_dir=None, video_output_path=None, kalman=True, mix_kalman=True):
        """
        Visualize keypoints on original frames and optionally save them as images or a video.
        """
        assert self.keypoints is not None and self.original_frames is not None, "Run `infer_keypoints` first."

        os.makedirs(output_dir, exist_ok=True) if output_dir else None

        height, width, _ = self.original_frames[0].shape
        if video_output_path:
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))
            
        if mix_kalman:
            smoothed_keypoints = kalman_filter_with_confidence(self.keypoints, self.confidences)
            reversed_confidences = self.confidences[::-1,:]
            re_smoothed_keypoints = kalman_filter_with_confidence(smoothed_keypoints[::-1,:,:], reversed_confidences)
            # smoothed_keypoints = (smoothed_keypoints + re_smoothed_keypoints[::-1,:,:])*0.5
            smoothed_keypoints = re_smoothed_keypoints[::-1, :, :]
            final_keypoints = mixed_keypoints_weighted(self.keypoints, self.confidences, smoothed_keypoints)
            final_keypoints = kalman_filter_without_confidence(final_keypoints)
        elif kalman:
            final_keypoints = kalman_filter_with_confidence(self.keypoints, self.confidences)
        else:
            final_keypoints = self.keypoints
        for i, frame in enumerate(tqdm(self.original_frames, desc="Visualizing keypoints")):
            frame_vis = frame.copy()
            for j, joint in enumerate(final_keypoints[i]):
                x, y = int(joint[0] * frame.shape[1]), int(joint[1] * frame.shape[0])  # Map normalized coords back
                color = (0, 255, 0)
                cv2.circle(frame_vis, (x, y), 5, color, -1)

            frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)

            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frame_vis)

            if video_output_path:
                out.write(frame_vis)

        if video_output_path:
            out.release()

    def process_video(self, video_array, output_dir=None, video_output_path=None):
        """
        Process the video to infer keypoints and optionally save visualizations.
        """
        self.infer_keypoints(video_array)
        self.visualize_keypoints(output_dir=output_dir, video_output_path=video_output_path)
        return self.keypoints, self.confidences


# Example usage
if __name__ == "__main__":
    
    # video_path = "test_video.mp4"
    model_path = "./ckpts/best_model_trace.pt"
    processor = VideoKeypointProcessor(model_path)
    # video_array = skvideo.io.vread(video_path)
    # print(video_array.shape) # (86, 612, 612, 3)
    # # keypoints, confidences = processor.process_video(video_path, video_output_path='out_video.avi')
    # cropped_keypoint, cropped_edge, cropped_fusion = processor.infer_heatmaps(video_array)
    # print(cropped_fusion.shape)
    # # cropped_fusion: [N, H, W]
    
    # # heatmaps转tensor的用法
    # # heatmaps_tensor = torch.from_numpy(cropped_fusion)
    # # heatmaps_tensor = heatmaps_tensor.unsqueeze(1)
    # # resized_tensor = F.interpolate(heatmaps_tensor, size=(56, 56), mode='bilinear', align_corners=False) # 120, 1, 56, 56
    # # resized_tensor = resized_tensor.repeat(1, 3, 1, 1) # 120,3,56,56 dtype=torch.float32
    
    
    # # heatmaps的可视化
    # heatmaps = (cropped_fusion * 255).astype(np.uint8)[..., np.newaxis]
    # # 视频输出路径和设置
    # output_path = "test_api_npy.mp4"
    # fps = 30  # 设置帧率为 30fps
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    
    # height, width = heatmaps.shape[1], heatmaps.shape[2]  # 高度和宽度来自 heatmaps 的形状
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)  # isColor=False 处理灰度图像

    # # 将每一帧写入视频
    # for frame in heatmaps:
    #     out.write(frame.astype(np.uint8))  # 将每一帧写入视频

    # # 释放资源
    # out.release()

    # Use multiprocess to extract heatmap without kalman
    # Single processor version for processing videos
    def process_single_video(obj, video_path, output_root, kalman, normal_kalman, selected_index, output_size=None):
        base_name = os.path.basename(video_path).split(".avi")[0]
        save_np = f'{base_name}.npy'
        os.makedirs(output_root, exist_ok=True)
        save_path = os.path.join(output_root, save_np)
        video_array = skvideo.io.vread(video_path)
        # Extract heatmap data
        data = obj.infer_heatmaps(video_array, kalman, normal_kalman)[selected_index]
        
        # Resize if needed
        if output_size is not None:
            resized_data = np.zeros((data.shape[0], output_size[0], output_size[1]), dtype=np.float32)
            for i, frame in enumerate(data):
                resized_data[i] = cv2.resize(frame, output_size)
            data = resized_data
            
        # Save the processed data
        np.save(save_path, data)
        return save_path
    
    # Process all videos in sequence
    def process_all_videos(obj, input_root, input_file, output_root, kalman, normal_kalman,
                          selected_index, resume=True, output_size=None):
        # Read video list from file
        with open(input_file, 'r') as f:
            data = f.readlines()
        
        videos = [os.path.join(input_root, f'{i.split(",")[0].strip()}.avi') for i in data if i.strip() != '']
        print(f'Total {len(videos)} videos to process.')
        
        results = []
        for video in tqdm(videos, desc="Processing heatmaps"):
            # Skip if file exists and resume is enabled
            save_path = os.path.join(output_root, f'{os.path.basename(video).split(".avi")[0]}.npy')
            if resume and os.path.exists(save_path):
                try:
                    np.load(save_path)
                    results.append(None)
                    continue
                except Exception:
                    pass
            
            # Process the video
            result = process_single_video(obj, video, output_root, kalman, normal_kalman, selected_index, output_size)
            results.append(result)
        
        return results
    
    # Process videos
    results = process_all_videos(
        obj=processor,
        input_root='data/swallow/stage_2',
        input_file='tmp/stage2_trainval.txt',
        output_root='data/swallow/swallow_heatmap56_sigma4',
        kalman=True,
        normal_kalman=False,
        selected_index=-1, # fusion heatmap
        output_size=(56,56),
    )
    