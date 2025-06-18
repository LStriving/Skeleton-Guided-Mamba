import json
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
                generate_a_heatmap(arr, start[None], start_value[None],sigma=sigma)
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
    def __init__(self, model_path, image_width=192, image_height=256, batch_size=32, num_workers=4, sigma=0.6, crop_mode='auto',device='cuda'):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sigma = sigma

        # Load model
        self.model = torch.jit.load(model_path).to(device)
        self.model.eval()
        self.device = device
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


    def _resize_video_frames(self, video_path):
        """
        Resize all frames of a video to the specified width and height.
        Also returns original frames for visualization.
        """
        # 使用 skvideo 读取视频
        videodata = skvideo.io.vread(video_path)

        resized_frames = []
        original_frames = []

        # 遍历每一帧并调整大小
        for frame in tqdm(videodata, desc="Resizing frames"):
            frame_np = frame  # skvideo 返回的已经是 numpy 数组
            original_frames.append(frame_np)  # 保留原始帧用于可视化
            resized_frame = cv2.resize(frame_np, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)

        return resized_frames, original_frames

    def infer_keypoints(self, video_path, kalman=True, normal_kalman=False):
        """
        Perform inference on the video and store keypoints and confidences.
        """
        resized_frames, self.original_frames = self._resize_video_frames(video_path)
        dataset = [self.preprocess(frame) for frame in resized_frames]
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        outputs_list = []
        for batch in tqdm(data_loader, desc="Inferencing keypoints"):
            batch = batch.to(self.device)
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
        if isinstance(kalman, str):
            kalman = eval(kalman)
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
        elif not kalman:
            return self.keypoints, confidences, h, w

        return self.keypoints, self.confidences, h, w
    
    def infer_heatmaps(self, video_path, kalman=True, normal_kalman=False):
        keypoints, confidences, h, w = self.infer_keypoints(video_path, kalman=kalman, normal_kalman=normal_kalman)
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

    def process_video(self, video_path, output_dir=None, video_output_path=None):
        """
        Process the video to infer keypoints and optionally save visualizations.
        """
        self.infer_keypoints(video_path)
        self.visualize_keypoints(output_dir=output_dir, video_output_path=video_output_path)
        return self.keypoints, self.confidences

class HumanKeypointProcessor:
    '''
    Class for deal with extracted human keypoints and return heatmaps
    '''
    def __init__(self, pred_root, video_root, skeleton, splits=['train','val','test'], sigma=0.6, crop_mode='auto', 
                 video_ext='.mp4',frame_height=1080,frame_width=900,skeleton_num=133):
        self.pred_root = pred_root
        self.pred_path = os.listdir(pred_root)
        self.pred_path = [os.path.join(pred_root, i) for i in self.pred_path if '.json' in i and self.contains(i, splits)]
        self.video_root = video_root
        self.split = splits
        if video_root is not None:
            self.videos = []
            for split in self.split:
                videos = os.listdir(os.path.join(video_root, split))
                videos = [os.path.join(video_root, split, i) for i in videos if video_ext in i]
                self.videos.extend(videos)
        self.sigma = sigma
        self.crop_mode = crop_mode
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.skeleton_num = skeleton_num 
        self.skeleton = skeleton if skeleton is not None else \
            np.array(self._get_mapping())

    @staticmethod
    def contains(long_str, inlist):
        for item in inlist:
            if item in long_str:
                return True
        return False
    
    def _get_mapping(self):
        keypoint_info={
            0:
            dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
            1:
            dict(
                name='left_eye',
                id=1,
                color=[51, 153, 255],
                type='upper',
                swap='right_eye'),
            2:
            dict(
                name='right_eye',
                id=2,
                color=[51, 153, 255],
                type='upper',
                swap='left_eye'),
            3:
            dict(
                name='left_ear',
                id=3,
                color=[51, 153, 255],
                type='upper',
                swap='right_ear'),
            4:
            dict(
                name='right_ear',
                id=4,
                color=[51, 153, 255],
                type='upper',
                swap='left_ear'),
            5:
            dict(
                name='left_shoulder',
                id=5,
                color=[0, 255, 0],
                type='upper',
                swap='right_shoulder'),
            6:
            dict(
                name='right_shoulder',
                id=6,
                color=[255, 128, 0],
                type='upper',
                swap='left_shoulder'),
            7:
            dict(
                name='left_elbow',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap='right_elbow'),
            8:
            dict(
                name='right_elbow',
                id=8,
                color=[255, 128, 0],
                type='upper',
                swap='left_elbow'),
            9:
            dict(
                name='left_wrist',
                id=9,
                color=[0, 255, 0],
                type='upper',
                swap='right_wrist'),
            10:
            dict(
                name='right_wrist',
                id=10,
                color=[255, 128, 0],
                type='upper',
                swap='left_wrist'),
            11:
            dict(
                name='left_hip',
                id=11,
                color=[0, 255, 0],
                type='lower',
                swap='right_hip'),
            12:
            dict(
                name='right_hip',
                id=12,
                color=[255, 128, 0],
                type='lower',
                swap='left_hip'),
            13:
            dict(
                name='left_knee',
                id=13,
                color=[0, 255, 0],
                type='lower',
                swap='right_knee'),
            14:
            dict(
                name='right_knee',
                id=14,
                color=[255, 128, 0],
                type='lower',
                swap='left_knee'),
            15:
            dict(
                name='left_ankle',
                id=15,
                color=[0, 255, 0],
                type='lower',
                swap='right_ankle'),
            16:
            dict(
                name='right_ankle',
                id=16,
                color=[255, 128, 0],
                type='lower',
                swap='left_ankle'),
            17:
            dict(
                name='left_big_toe',
                id=17,
                color=[255, 128, 0],
                type='lower',
                swap='right_big_toe'),
            18:
            dict(
                name='left_small_toe',
                id=18,
                color=[255, 128, 0],
                type='lower',
                swap='right_small_toe'),
            19:
            dict(
                name='left_heel',
                id=19,
                color=[255, 128, 0],
                type='lower',
                swap='right_heel'),
            20:
            dict(
                name='right_big_toe',
                id=20,
                color=[255, 128, 0],
                type='lower',
                swap='left_big_toe'),
            21:
            dict(
                name='right_small_toe',
                id=21,
                color=[255, 128, 0],
                type='lower',
                swap='left_small_toe'),
            22:
            dict(
                name='right_heel',
                id=22,
                color=[255, 128, 0],
                type='lower',
                swap='left_heel'),
            23:
            dict(
                name='face-0',
                id=23,
                color=[255, 255, 255],
                type='',
                swap='face-16'),
            24:
            dict(
                name='face-1',
                id=24,
                color=[255, 255, 255],
                type='',
                swap='face-15'),
            25:
            dict(
                name='face-2',
                id=25,
                color=[255, 255, 255],
                type='',
                swap='face-14'),
            26:
            dict(
                name='face-3',
                id=26,
                color=[255, 255, 255],
                type='',
                swap='face-13'),
            27:
            dict(
                name='face-4',
                id=27,
                color=[255, 255, 255],
                type='',
                swap='face-12'),
            28:
            dict(
                name='face-5',
                id=28,
                color=[255, 255, 255],
                type='',
                swap='face-11'),
            29:
            dict(
                name='face-6',
                id=29,
                color=[255, 255, 255],
                type='',
                swap='face-10'),
            30:
            dict(
                name='face-7',
                id=30,
                color=[255, 255, 255],
                type='',
                swap='face-9'),
            31:
            dict(name='face-8', id=31, color=[255, 255, 255], type='', swap=''),
            32:
            dict(
                name='face-9',
                id=32,
                color=[255, 255, 255],
                type='',
                swap='face-7'),
            33:
            dict(
                name='face-10',
                id=33,
                color=[255, 255, 255],
                type='',
                swap='face-6'),
            34:
            dict(
                name='face-11',
                id=34,
                color=[255, 255, 255],
                type='',
                swap='face-5'),
            35:
            dict(
                name='face-12',
                id=35,
                color=[255, 255, 255],
                type='',
                swap='face-4'),
            36:
            dict(
                name='face-13',
                id=36,
                color=[255, 255, 255],
                type='',
                swap='face-3'),
            37:
            dict(
                name='face-14',
                id=37,
                color=[255, 255, 255],
                type='',
                swap='face-2'),
            38:
            dict(
                name='face-15',
                id=38,
                color=[255, 255, 255],
                type='',
                swap='face-1'),
            39:
            dict(
                name='face-16',
                id=39,
                color=[255, 255, 255],
                type='',
                swap='face-0'),
            40:
            dict(
                name='face-17',
                id=40,
                color=[255, 255, 255],
                type='',
                swap='face-26'),
            41:
            dict(
                name='face-18',
                id=41,
                color=[255, 255, 255],
                type='',
                swap='face-25'),
            42:
            dict(
                name='face-19',
                id=42,
                color=[255, 255, 255],
                type='',
                swap='face-24'),
            43:
            dict(
                name='face-20',
                id=43,
                color=[255, 255, 255],
                type='',
                swap='face-23'),
            44:
            dict(
                name='face-21',
                id=44,
                color=[255, 255, 255],
                type='',
                swap='face-22'),
            45:
            dict(
                name='face-22',
                id=45,
                color=[255, 255, 255],
                type='',
                swap='face-21'),
            46:
            dict(
                name='face-23',
                id=46,
                color=[255, 255, 255],
                type='',
                swap='face-20'),
            47:
            dict(
                name='face-24',
                id=47,
                color=[255, 255, 255],
                type='',
                swap='face-19'),
            48:
            dict(
                name='face-25',
                id=48,
                color=[255, 255, 255],
                type='',
                swap='face-18'),
            49:
            dict(
                name='face-26',
                id=49,
                color=[255, 255, 255],
                type='',
                swap='face-17'),
            50:
            dict(name='face-27', id=50, color=[255, 255, 255], type='', swap=''),
            51:
            dict(name='face-28', id=51, color=[255, 255, 255], type='', swap=''),
            52:
            dict(name='face-29', id=52, color=[255, 255, 255], type='', swap=''),
            53:
            dict(name='face-30', id=53, color=[255, 255, 255], type='', swap=''),
            54:
            dict(
                name='face-31',
                id=54,
                color=[255, 255, 255],
                type='',
                swap='face-35'),
            55:
            dict(
                name='face-32',
                id=55,
                color=[255, 255, 255],
                type='',
                swap='face-34'),
            56:
            dict(name='face-33', id=56, color=[255, 255, 255], type='', swap=''),
            57:
            dict(
                name='face-34',
                id=57,
                color=[255, 255, 255],
                type='',
                swap='face-32'),
            58:
            dict(
                name='face-35',
                id=58,
                color=[255, 255, 255],
                type='',
                swap='face-31'),
            59:
            dict(
                name='face-36',
                id=59,
                color=[255, 255, 255],
                type='',
                swap='face-45'),
            60:
            dict(
                name='face-37',
                id=60,
                color=[255, 255, 255],
                type='',
                swap='face-44'),
            61:
            dict(
                name='face-38',
                id=61,
                color=[255, 255, 255],
                type='',
                swap='face-43'),
            62:
            dict(
                name='face-39',
                id=62,
                color=[255, 255, 255],
                type='',
                swap='face-42'),
            63:
            dict(
                name='face-40',
                id=63,
                color=[255, 255, 255],
                type='',
                swap='face-47'),
            64:
            dict(
                name='face-41',
                id=64,
                color=[255, 255, 255],
                type='',
                swap='face-46'),
            65:
            dict(
                name='face-42',
                id=65,
                color=[255, 255, 255],
                type='',
                swap='face-39'),
            66:
            dict(
                name='face-43',
                id=66,
                color=[255, 255, 255],
                type='',
                swap='face-38'),
            67:
            dict(
                name='face-44',
                id=67,
                color=[255, 255, 255],
                type='',
                swap='face-37'),
            68:
            dict(
                name='face-45',
                id=68,
                color=[255, 255, 255],
                type='',
                swap='face-36'),
            69:
            dict(
                name='face-46',
                id=69,
                color=[255, 255, 255],
                type='',
                swap='face-41'),
            70:
            dict(
                name='face-47',
                id=70,
                color=[255, 255, 255],
                type='',
                swap='face-40'),
            71:
            dict(
                name='face-48',
                id=71,
                color=[255, 255, 255],
                type='',
                swap='face-54'),
            72:
            dict(
                name='face-49',
                id=72,
                color=[255, 255, 255],
                type='',
                swap='face-53'),
            73:
            dict(
                name='face-50',
                id=73,
                color=[255, 255, 255],
                type='',
                swap='face-52'),
            74:
            dict(name='face-51', id=74, color=[255, 255, 255], type='', swap=''),
            75:
            dict(
                name='face-52',
                id=75,
                color=[255, 255, 255],
                type='',
                swap='face-50'),
            76:
            dict(
                name='face-53',
                id=76,
                color=[255, 255, 255],
                type='',
                swap='face-49'),
            77:
            dict(
                name='face-54',
                id=77,
                color=[255, 255, 255],
                type='',
                swap='face-48'),
            78:
            dict(
                name='face-55',
                id=78,
                color=[255, 255, 255],
                type='',
                swap='face-59'),
            79:
            dict(
                name='face-56',
                id=79,
                color=[255, 255, 255],
                type='',
                swap='face-58'),
            80:
            dict(name='face-57', id=80, color=[255, 255, 255], type='', swap=''),
            81:
            dict(
                name='face-58',
                id=81,
                color=[255, 255, 255],
                type='',
                swap='face-56'),
            82:
            dict(
                name='face-59',
                id=82,
                color=[255, 255, 255],
                type='',
                swap='face-55'),
            83:
            dict(
                name='face-60',
                id=83,
                color=[255, 255, 255],
                type='',
                swap='face-64'),
            84:
            dict(
                name='face-61',
                id=84,
                color=[255, 255, 255],
                type='',
                swap='face-63'),
            85:
            dict(name='face-62', id=85, color=[255, 255, 255], type='', swap=''),
            86:
            dict(
                name='face-63',
                id=86,
                color=[255, 255, 255],
                type='',
                swap='face-61'),
            87:
            dict(
                name='face-64',
                id=87,
                color=[255, 255, 255],
                type='',
                swap='face-60'),
            88:
            dict(
                name='face-65',
                id=88,
                color=[255, 255, 255],
                type='',
                swap='face-67'),
            89:
            dict(name='face-66', id=89, color=[255, 255, 255], type='', swap=''),
            90:
            dict(
                name='face-67',
                id=90,
                color=[255, 255, 255],
                type='',
                swap='face-65'),
            91:
            dict(
                name='left_hand_root',
                id=91,
                color=[255, 255, 255],
                type='',
                swap='right_hand_root'),
            92:
            dict(
                name='left_thumb1',
                id=92,
                color=[255, 128, 0],
                type='',
                swap='right_thumb1'),
            93:
            dict(
                name='left_thumb2',
                id=93,
                color=[255, 128, 0],
                type='',
                swap='right_thumb2'),
            94:
            dict(
                name='left_thumb3',
                id=94,
                color=[255, 128, 0],
                type='',
                swap='right_thumb3'),
            95:
            dict(
                name='left_thumb4',
                id=95,
                color=[255, 128, 0],
                type='',
                swap='right_thumb4'),
            96:
            dict(
                name='left_forefinger1',
                id=96,
                color=[255, 153, 255],
                type='',
                swap='right_forefinger1'),
            97:
            dict(
                name='left_forefinger2',
                id=97,
                color=[255, 153, 255],
                type='',
                swap='right_forefinger2'),
            98:
            dict(
                name='left_forefinger3',
                id=98,
                color=[255, 153, 255],
                type='',
                swap='right_forefinger3'),
            99:
            dict(
                name='left_forefinger4',
                id=99,
                color=[255, 153, 255],
                type='',
                swap='right_forefinger4'),
            100:
            dict(
                name='left_middle_finger1',
                id=100,
                color=[102, 178, 255],
                type='',
                swap='right_middle_finger1'),
            101:
            dict(
                name='left_middle_finger2',
                id=101,
                color=[102, 178, 255],
                type='',
                swap='right_middle_finger2'),
            102:
            dict(
                name='left_middle_finger3',
                id=102,
                color=[102, 178, 255],
                type='',
                swap='right_middle_finger3'),
            103:
            dict(
                name='left_middle_finger4',
                id=103,
                color=[102, 178, 255],
                type='',
                swap='right_middle_finger4'),
            104:
            dict(
                name='left_ring_finger1',
                id=104,
                color=[255, 51, 51],
                type='',
                swap='right_ring_finger1'),
            105:
            dict(
                name='left_ring_finger2',
                id=105,
                color=[255, 51, 51],
                type='',
                swap='right_ring_finger2'),
            106:
            dict(
                name='left_ring_finger3',
                id=106,
                color=[255, 51, 51],
                type='',
                swap='right_ring_finger3'),
            107:
            dict(
                name='left_ring_finger4',
                id=107,
                color=[255, 51, 51],
                type='',
                swap='right_ring_finger4'),
            108:
            dict(
                name='left_pinky_finger1',
                id=108,
                color=[0, 255, 0],
                type='',
                swap='right_pinky_finger1'),
            109:
            dict(
                name='left_pinky_finger2',
                id=109,
                color=[0, 255, 0],
                type='',
                swap='right_pinky_finger2'),
            110:
            dict(
                name='left_pinky_finger3',
                id=110,
                color=[0, 255, 0],
                type='',
                swap='right_pinky_finger3'),
            111:
            dict(
                name='left_pinky_finger4',
                id=111,
                color=[0, 255, 0],
                type='',
                swap='right_pinky_finger4'),
            112:
            dict(
                name='right_hand_root',
                id=112,
                color=[255, 255, 255],
                type='',
                swap='left_hand_root'),
            113:
            dict(
                name='right_thumb1',
                id=113,
                color=[255, 128, 0],
                type='',
                swap='left_thumb1'),
            114:
            dict(
                name='right_thumb2',
                id=114,
                color=[255, 128, 0],
                type='',
                swap='left_thumb2'),
            115:
            dict(
                name='right_thumb3',
                id=115,
                color=[255, 128, 0],
                type='',
                swap='left_thumb3'),
            116:
            dict(
                name='right_thumb4',
                id=116,
                color=[255, 128, 0],
                type='',
                swap='left_thumb4'),
            117:
            dict(
                name='right_forefinger1',
                id=117,
                color=[255, 153, 255],
                type='',
                swap='left_forefinger1'),
            118:
            dict(
                name='right_forefinger2',
                id=118,
                color=[255, 153, 255],
                type='',
                swap='left_forefinger2'),
            119:
            dict(
                name='right_forefinger3',
                id=119,
                color=[255, 153, 255],
                type='',
                swap='left_forefinger3'),
            120:
            dict(
                name='right_forefinger4',
                id=120,
                color=[255, 153, 255],
                type='',
                swap='left_forefinger4'),
            121:
            dict(
                name='right_middle_finger1',
                id=121,
                color=[102, 178, 255],
                type='',
                swap='left_middle_finger1'),
            122:
            dict(
                name='right_middle_finger2',
                id=122,
                color=[102, 178, 255],
                type='',
                swap='left_middle_finger2'),
            123:
            dict(
                name='right_middle_finger3',
                id=123,
                color=[102, 178, 255],
                type='',
                swap='left_middle_finger3'),
            124:
            dict(
                name='right_middle_finger4',
                id=124,
                color=[102, 178, 255],
                type='',
                swap='left_middle_finger4'),
            125:
            dict(
                name='right_ring_finger1',
                id=125,
                color=[255, 51, 51],
                type='',
                swap='left_ring_finger1'),
            126:
            dict(
                name='right_ring_finger2',
                id=126,
                color=[255, 51, 51],
                type='',
                swap='left_ring_finger2'),
            127:
            dict(
                name='right_ring_finger3',
                id=127,
                color=[255, 51, 51],
                type='',
                swap='left_ring_finger3'),
            128:
            dict(
                name='right_ring_finger4',
                id=128,
                color=[255, 51, 51],
                type='',
                swap='left_ring_finger4'),
            129:
            dict(
                name='right_pinky_finger1',
                id=129,
                color=[0, 255, 0],
                type='',
                swap='left_pinky_finger1'),
            130:
            dict(
                name='right_pinky_finger2',
                id=130,
                color=[0, 255, 0],
                type='',
                swap='left_pinky_finger2'),
            131:
            dict(
                name='right_pinky_finger3',
                id=131,
                color=[0, 255, 0],
                type='',
                swap='left_pinky_finger3'),
            132:
            dict(
                name='right_pinky_finger4',
                id=132,
                color=[0, 255, 0],
                type='',
                swap='left_pinky_finger4')
        },
        skeleton_info={
            0:
            dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
            1:
            dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
            2:
            dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
            3:
            dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
            4:
            dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
            5:
            dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
            6:
            dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
            7:
            dict(
                link=('left_shoulder', 'right_shoulder'),
                id=7,
                color=[51, 153, 255]),
            8:
            dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
            9:
            dict(
                link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
            10:
            dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
            11:
            dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
            12:
            dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
            13:
            dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
            14:
            dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
            15:
            dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
            16:
            dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
            17:
            dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
            18:
            dict(
                link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
            19:
            dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
            20:
            dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
            21:
            dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
            22:
            dict(
                link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
            23:
            dict(
                link=('right_ankle', 'right_small_toe'),
                id=23,
                color=[255, 128, 0]),
            24:
            dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
            25:
            dict(
                link=('left_hand_root', 'left_thumb1'), id=25, color=[255, 128,
                                                                    0]),
            26:
            dict(link=('left_thumb1', 'left_thumb2'), id=26, color=[255, 128, 0]),
            27:
            dict(link=('left_thumb2', 'left_thumb3'), id=27, color=[255, 128, 0]),
            28:
            dict(link=('left_thumb3', 'left_thumb4'), id=28, color=[255, 128, 0]),
            29:
            dict(
                link=('left_hand_root', 'left_forefinger1'),
                id=29,
                color=[255, 153, 255]),
            30:
            dict(
                link=('left_forefinger1', 'left_forefinger2'),
                id=30,
                color=[255, 153, 255]),
            31:
            dict(
                link=('left_forefinger2', 'left_forefinger3'),
                id=31,
                color=[255, 153, 255]),
            32:
            dict(
                link=('left_forefinger3', 'left_forefinger4'),
                id=32,
                color=[255, 153, 255]),
            33:
            dict(
                link=('left_hand_root', 'left_middle_finger1'),
                id=33,
                color=[102, 178, 255]),
            34:
            dict(
                link=('left_middle_finger1', 'left_middle_finger2'),
                id=34,
                color=[102, 178, 255]),
            35:
            dict(
                link=('left_middle_finger2', 'left_middle_finger3'),
                id=35,
                color=[102, 178, 255]),
            36:
            dict(
                link=('left_middle_finger3', 'left_middle_finger4'),
                id=36,
                color=[102, 178, 255]),
            37:
            dict(
                link=('left_hand_root', 'left_ring_finger1'),
                id=37,
                color=[255, 51, 51]),
            38:
            dict(
                link=('left_ring_finger1', 'left_ring_finger2'),
                id=38,
                color=[255, 51, 51]),
            39:
            dict(
                link=('left_ring_finger2', 'left_ring_finger3'),
                id=39,
                color=[255, 51, 51]),
            40:
            dict(
                link=('left_ring_finger3', 'left_ring_finger4'),
                id=40,
                color=[255, 51, 51]),
            41:
            dict(
                link=('left_hand_root', 'left_pinky_finger1'),
                id=41,
                color=[0, 255, 0]),
            42:
            dict(
                link=('left_pinky_finger1', 'left_pinky_finger2'),
                id=42,
                color=[0, 255, 0]),
            43:
            dict(
                link=('left_pinky_finger2', 'left_pinky_finger3'),
                id=43,
                color=[0, 255, 0]),
            44:
            dict(
                link=('left_pinky_finger3', 'left_pinky_finger4'),
                id=44,
                color=[0, 255, 0]),
            45:
            dict(
                link=('right_hand_root', 'right_thumb1'),
                id=45,
                color=[255, 128, 0]),
            46:
            dict(
                link=('right_thumb1', 'right_thumb2'), id=46, color=[255, 128, 0]),
            47:
            dict(
                link=('right_thumb2', 'right_thumb3'), id=47, color=[255, 128, 0]),
            48:
            dict(
                link=('right_thumb3', 'right_thumb4'), id=48, color=[255, 128, 0]),
            49:
            dict(
                link=('right_hand_root', 'right_forefinger1'),
                id=49,
                color=[255, 153, 255]),
            50:
            dict(
                link=('right_forefinger1', 'right_forefinger2'),
                id=50,
                color=[255, 153, 255]),
            51:
            dict(
                link=('right_forefinger2', 'right_forefinger3'),
                id=51,
                color=[255, 153, 255]),
            52:
            dict(
                link=('right_forefinger3', 'right_forefinger4'),
                id=52,
                color=[255, 153, 255]),
            53:
            dict(
                link=('right_hand_root', 'right_middle_finger1'),
                id=53,
                color=[102, 178, 255]),
            54:
            dict(
                link=('right_middle_finger1', 'right_middle_finger2'),
                id=54,
                color=[102, 178, 255]),
            55:
            dict(
                link=('right_middle_finger2', 'right_middle_finger3'),
                id=55,
                color=[102, 178, 255]),
            56:
            dict(
                link=('right_middle_finger3', 'right_middle_finger4'),
                id=56,
                color=[102, 178, 255]),
            57:
            dict(
                link=('right_hand_root', 'right_ring_finger1'),
                id=57,
                color=[255, 51, 51]),
            58:
            dict(
                link=('right_ring_finger1', 'right_ring_finger2'),
                id=58,
                color=[255, 51, 51]),
            59:
            dict(
                link=('right_ring_finger2', 'right_ring_finger3'),
                id=59,
                color=[255, 51, 51]),
            60:
            dict(
                link=('right_ring_finger3', 'right_ring_finger4'),
                id=60,
                color=[255, 51, 51]),
            61:
            dict(
                link=('right_hand_root', 'right_pinky_finger1'),
                id=61,
                color=[0, 255, 0]),
            62:
            dict(
                link=('right_pinky_finger1', 'right_pinky_finger2'),
                id=62,
                color=[0, 255, 0]),
            63:
            dict(
                link=('right_pinky_finger2', 'right_pinky_finger3'),
                id=63,
                color=[0, 255, 0]),
            64:
            dict(
                link=('right_pinky_finger3', 'right_pinky_finger4'),
                id=64,
                color=[0, 255, 0])
        },
        # joint_weights=[1.] * 133,
        name_to_id = {}
        # print(keypoint_info[0])
        for k, v in keypoint_info[0].items():
            name_to_id[v['name']] = v['id']
        return [(name_to_id[s], name_to_id[e]) for s, e in 
            (skel['link'] for skel in skeleton_info[0].values())]


    def get_single_video_keypoints(self, video_name_or_index, kalman=False):
        if isinstance(video_name_or_index, int):
            pred_path = self.pred_path[video_name_or_index]
        elif isinstance(video_name_or_index, str):
            pred_name = f'{video_name_or_index}.json'
            pred_path = None
            for item in self.pred_path:
                if pred_name in item:
                    pred_path = item
                    break
        if pred_path is None:
            raise ValueError(f'{video_name_or_index} not found.')
        
        with open(pred_path, 'r') as f:
            prediction = json.load(f)
        # json2np
        if not kalman:
            return self._json2np(prediction)
        else:
            keypoints, confidences = self._json2np(prediction)
            keypoints[:,:,0] /= self.frame_width
            keypoints[:,:,1] /= self.frame_height
            if len(keypoints) > 1:
                smoothed_keypoints = kalman_filter_with_confidence(keypoints, confidences)
                reversed_confidences = confidences[::-1,:]
                re_smoothed_keypoints = kalman_filter_with_confidence(smoothed_keypoints[::-1,:,:], reversed_confidences)
                # smoothed_keypoints = (smoothed_keypoints + re_smoothed_keypoints[::-1,:,:])*0.5
                smoothed_keypoints = re_smoothed_keypoints[::-1, :, :]
                final_keypoints = mixed_keypoints_weighted(keypoints, confidences, smoothed_keypoints)
                final_keypoints = kalman_filter_without_confidence(final_keypoints)
                final_keypoints[:,:,0] *= self.frame_width
                final_keypoints[:,:,1] *= self.frame_height
                return final_keypoints, confidences
    
    def _json2np(self, json_data):
        total_frame = len(json_data)
        # create numpy data
        keypoints = np.zeros((total_frame, self.skeleton_num, 2))
        keypoint_scores = np.zeros((total_frame, self.skeleton_num))
        for frame_pred in json_data:
            frame_id = int(frame_pred['frame_id'])
            keypoint_scores[frame_id] = np.array(frame_pred['instances'][0]['keypoint_scores'])
            keypoints[frame_id] = np.array(frame_pred['instances'][0]['keypoints'])
        return keypoints, keypoint_scores

    def export_dataset_heatmap(self, output_root, kalman=False, saved_type='fusion', resume=True, num_workers=8):
        """
        Export dataset heatmaps with multiprocessing support
        
        Args:
            output_root: Root directory to save the exported data
            kalman: Whether to use Kalman filtering
            saved_type: Type of data to save ('fusion', 'line', or 'keypoints')
            resume: Whether to resume from previous export
            num_workers: Number of parallel processes to use
        """
        if saved_type == 'fusion':
            selected_index = -1
        elif saved_type == 'line':
            selected_index = 1
        elif saved_type == 'keypoints':
            selected_index = 0
        else:
            raise ValueError(f'saved_type should be in [fusion, line, keypoints] rather than {saved_type}')
        
        # Create the output directory with sigma value
        output_dir = os.path.join(output_root, f'simga_{self.sigma}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the worker function for multiprocessing
        def process_video(idx):
            pred_name = self.pred_path[idx]
            base_name = os.path.basename(pred_name).split(".")[0]
            save_np = os.path.join(output_dir, f'{base_name}.npy')
            
            if resume and os.path.exists(save_np):
                return None
            
            try:
                data = self.get_heatmap(idx, kalman)[selected_index]
                np.save(save_np, data)
                return base_name
            except Exception as e:
                return f"Error processing {base_name}: {str(e)}"
        
        # Use multiprocessing to process videos in parallel
        import multiprocessing as mp
        from tqdm.auto import tqdm
        
        # Create a pool of workers
        pool = mp.Pool(processes=num_workers)
        
        # Submit all tasks and track with tqdm
        results = list(tqdm(
            pool.imap(process_video, range(len(self.pred_path))),
            total=len(self.pred_path),
            desc="Exporting heatmaps"
        ))
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Report any errors
        errors = [r for r in results if isinstance(r, str) and r.startswith("Error")]
        if errors:
            print(f"Encountered {len(errors)} errors during processing:")
            for err in errors[:10]:  # Show first 10 errors
                print(f"  {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

    def get_heatmap(self, video_name_or_index, kalman=False):
        keypoints, confidences = self.get_single_video_keypoints(video_name_or_index, kalman)
        h, w = self.frame_height, self.frame_width
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
        
# Example usage
if __name__ == "__main__":
    
    # video_path = "test_video.mp4"
    model_path = "./pretrained/heatmap/best_model_trace.pt"
    processor = VideoKeypointProcessor(model_path, sigma=4)
    # cropped_keypoint, cropped_edge, cropped_fusion = processor.infer_heatmaps(video_path)
    # # cropped_fusion: [N, H, W]
    
    # heatmaps转tensor的用法
    # heatmaps_tensor = torch.from_numpy(cropped_fusion)
    # heatmaps_tensor = heatmaps_tensor.unsqueeze(1)
    # resized_tensor = F.interpolate(heatmaps_tensor, size=(56, 56), mode='bilinear', align_corners=False) # 120, 1, 56, 56
    # resized_tensor = resized_tensor.repeat(1, 3, 1, 1) # 120,3,56,56 dtype=torch.float32
    
    
    # heatmaps的可视化
    # heatmaps = (cropped_fusion * 255).astype(np.uint8)[..., np.newaxis]
    # # 视频输出路径和设置
    # output_path = "test_api_video.mp4"
    # fps = 30  # 设置帧率为 30fps
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    
    # height, width = heatmaps.shape[1], heatmaps.shape[2]  # 高度和宽度来自 heatmaps 的形状
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)  # isColor=False 处理灰度图像

    # # 将每一帧写入视频
    # for frame in heatmaps:
    #     out.write(frame.astype(np.uint8))  # 将每一帧写入视频

    # # 释放资源
    # out.release()
    # from multiprocessing import Pool
    # processer = HumanKeypointProcessor(
    #     pred_root='/mnt/cephfs/home/liyirui/project/mmpose/predictions',
    #     video_root=None,
    #     splits=['val','test'],
    #     skeleton=None,
    #     sigma=1
    # )
    # def process_item(args):
    #     id, obj, output_root, kalman, selected_index, resume, output_size = args
    #     pred_name = obj.pred_path[id]
    #     base_name = os.path.basename(pred_name).split(".")[0]
    #     save_np = f'{base_name}.npy'
    #     output_dir = os.path.join(output_root, f'sigma_{obj.sigma}')
    #     os.makedirs(output_dir, exist_ok=True)
    #     save_path = os.path.join(output_dir, save_np)
        
    #     if resume and os.path.exists(save_path):
    #         # try to open it
    #         try:
    #             np.load(save_path)
    #             return None
    #         except Exception as e:
    #             ...
        
    #     data = obj.get_heatmap(id, kalman)[selected_index]
    #     if output_size is not None:
    #         # resize data
    #         data = resize(data, output_size)
    #     np.save(save_path, data)
    #     return save_path
    
    # def resize(numpy_data, output_size):
    #     """
    #     Resize the input numpy array to the specified output size.

    #     Args:
    #         numpy_data (np.ndarray): The input array to resize.
    #         output_size (tuple): The desired output size (height, width).

    #     Returns:
    #         np.ndarray: The resized array.
    #     """
    #     from PIL import Image

    #     # 将 NumPy 数组转换为 PIL 图像
    #     image = Image.fromarray((numpy_data * 255).astype(np.uint8))
    #     # 调整图像大小
    #     resized_image = image.resize(output_size, Image.ANTIALIAS)
    #     # 将调整大小后的图像转换回 NumPy 数组
    #     resized_data = np.array(resized_image) / 255.0  # 归一化到 [0, 1] 范围
    #     return resized_data

    # def run_parallel_processing(obj, output_root, kalman, selected_index, resume=True, num_processes=None, output_size=None):
    #     if num_processes is None:
    #         num_processes = os.cpu_count()  # Use all available cores
        
    #     with Pool(processes=num_processes) as pool:
    #         args = [(i, obj, output_root, kalman, selected_index, resume, output_size) 
    #                 for i in range(len(obj.pred_path))]
            
    #         results = list(tqdm(
    #             pool.imap(process_item, args),
    #             total=len(obj.pred_path),
    #             desc="Processing heatmaps"
    #         ))
        
    #     return results
    # # export 
    # results = run_parallel_processing(
    #     obj=processer,
    #     output_root='/mnt/cephfs/dataset/MMA-52/heatmap',
    #     kalman=False,
    #     selected_index=-1,
    # )
    def process_single_video(obj, video_path, output_root, kalman, normal_kalman, selected_index, output_size=None):
        base_name = os.path.basename(video_path).split(".avi")[0]
        save_np = f'{base_name}.npy'
        os.makedirs(output_root, exist_ok=True)
        save_path = os.path.join(output_root, save_np)
        
        # Extract heatmap data
        data = obj.infer_heatmaps(video_path, kalman, normal_kalman)[selected_index]
        
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
        
        videos = [os.path.join(input_root, f'{i.strip()}.avi') for i in data if i.strip() != '']
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
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./pretrained/heatmap/best_model_trace.pt')
    parser.add_argument('--input_root', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_root', type=str)
    parser.add_argument('--kalman', type=str, default='True')
    parser.add_argument('--normal_kalman', action='store_true')
    parser.add_argument('--selected_index', type=int, default=-1)
    parser.add_argument('--output_size', type=tuple, default=(56,56))
    parser.add_argument('--sigma', type=float, default=4)
    args = parser.parse_args()
    
    processor = VideoKeypointProcessor(args.model_path, sigma=args.sigma)
    results = process_all_videos(
        obj=processor,
        input_root=args.input_root,
        input_file=args.input_file,
        output_root=args.output_root,
        kalman=args.kalman,
        normal_kalman=args.normal_kalman,
        selected_index=args.selected_index,
        output_size=args.output_size,
    )
