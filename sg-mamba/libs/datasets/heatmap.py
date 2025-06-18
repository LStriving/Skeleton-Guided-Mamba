import os
import io
import torch
import numpy as np
from tqdm import tqdm

import mmengine
from .datasets import register_dataset
from .data_utils import truncate_feats
from .swallow import SwallowDataset        


@register_dataset("heatmap_rawdata")
class HeatmapRawDataDataset(SwallowDataset):
    def __init__(self, 
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        feature_type="",
        two_stage=False, # two stage training
        stage_at=0,      # stage to start training
        desired_actions=None, # desired action label names):
        resize_to:int=None,  # resize the input features
        **kwargs
    ):
        super().__init__(is_training, split, feat_folder, json_file, feat_stride, num_frames, default_fps, downsample_rate, max_seq_len, trunc_thresh, crop_ratio, input_dim, num_classes, file_prefix, file_ext, force_upsampling, feature_type, two_stage, stage_at, desired_actions)
        self.resize_to = resize_to

    def __getitem__(self, index):
        video_item = self.data_list[index]

        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] +self.feature_type + self.file_ext)
        if "npy" in self.file_ext:
            data = io.BytesIO(mmengine.get(filename))
            feats = np.load(data).astype(np.float32)    # T, W, H
        elif "pt" in self.file_ext:
            data = io.BytesIO(mmengine.get(filename))
            feats = torch.load(data)
            feats = feats.numpy().astype(np.float32)
        else:
            raise NotImplementedError
        
        # resize the features
        if self.resize_to is not None and feats.shape[-1] != self.resize_to:
            feats = torch.from_numpy(feats)
            assert len(feats.shape) == 3, f"Invalid shape {feats.shape}"
            feats = torch.nn.functional.interpolate(feats.unsqueeze(1), (self.resize_to, self.resize_to), mode='bilinear', align_corners=False)
        else:
            # unsqueeze the feats
            feats = torch.from_numpy(feats).unsqueeze(1)  # T x C x H x W
        

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate

        feats = feats.transpose(0, 1).contiguous()  #  C x T x H x W

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T x H x W
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}
        
        # truncate the features during training
        if self.is_training and (segments is not None):
            max_seq_len = self.max_seq_len * feat_stride + self.num_frames
            data_dict = truncate_feats(
                data_dict, max_seq_len, self.trunc_thresh, self.crop_ratio
            )
        
        return data_dict
        