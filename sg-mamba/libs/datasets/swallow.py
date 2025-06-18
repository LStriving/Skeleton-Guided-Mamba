import os, tarfile
import io
import pickle
import json
from pprint import pprint
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import mmengine
import skvideo.io
from PIL import Image

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("swallow")
class SwallowDataset(Dataset):
    def __init__(
        self,
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
        desired_actions=None, # desired action label names
        **kwargs
    ):
        # file path
        # # assert os.path.exists(feat_folder) and os.path.exists(json_file)
        # assert client.isdir(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feature_type = feature_type
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        if isinstance(split, str):
            split = [split]
        if isinstance(split, list) or isinstance(split, tuple):
            split = [s.lower() for s in split]
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        
        # two stage training
        self.two_stage = two_stage
        if self.two_stage is True:
            assert stage_at in [1, 2]
        self.stage_at = stage_at
        self.desired_actions = desired_actions

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert label_dict is not None
        assert desired_actions is not None
        assert len(label_dict) == num_classes or \
                len(label_dict) == len(desired_actions), f'{len(label_dict)} vs {num_classes}, label dict: {label_dict}'
        self.data_list = dict_db
        self.label_dict = label_dict

        tiou_threshold = kwargs.get('tiou_threshold', np.linspace(0.1, 0.7, 7))
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'swallow',
            'tiou_thresholds': tiou_threshold,
            # we will mask out cliff diving
            'empty_label_ids': [],
        }
        print(f"SwallowDataset: {len(self.data_list)} videos loaded.")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        if 'database' in  json_data:
            json_data = json_data['database']
        json_db = json_data

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    if self.desired_actions is not None and act['label'] in self.desired_actions:
                        label_dict[act['label']] = act['label_id']
                    elif self.desired_actions is None:
                        label_dict[act['label']] = act['label_id']
                    elif act['label'] not in self.desired_actions:
                        continue
        # remap the label ids
        if self.desired_actions is not None:
            # remap the label dict ids
            label_dict = {k: i for i, k in enumerate(label_dict.keys())}

            for _, value in json_db.items():
                new_act = []
                for act in value['annotations']:
                    if act['label'] in self.desired_actions:
                        act['label_id'] = label_dict[act['label']]
                        new_act.append(act)
                    else:
                        continue
                value['annotations'] = new_act
            
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            # feat_file = os.path.join(self.feat_folder,
            #                          self.file_prefix + key + self.file_ext)
            # if not os.path.exists(feat_file):
            # if not client.contains(feat_file):
                # continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] +self.feature_type + self.file_ext)
        if "npy" in self.file_ext:
            data = io.BytesIO(mmengine.get(filename))
            feats = np.load(data).astype(np.float32)
        elif "pt" in self.file_ext:
            data = io.BytesIO(mmengine.get(filename))
            feats = torch.load(data)
            feats = feats.numpy().astype(np.float32)
        elif "npz" in self.file_ext:
            data = io.BytesIO(mmengine.get(filename))
            feat_spa = np.load(data)['feat_spa'].astype(np.float32) # C x T
            feat_tem = np.load(data)['feat_tem'].astype(np.float32) # C x T
            feats = np.concatenate([feat_spa,feat_tem],axis=0)
        else:
            raise NotImplementedError


        # i3d_feat_file = os.path.join("/mnt/petrelfs/chenguo/data/thumos/feature/th14_mae_g_16_4",self.file_prefix + video_item['id'] + ".npy")
        # i3d_feats  = np.load(i3d_feat_file).astype(np.float32)

        # feats = F.interpolate(torch.from_numpy(feats).permute(1,0).unsqueeze(0), size=i3d_feats.shape[0], mode='linear',align_corners=False)[0,...].permute(1,0)
        # feats = feats.numpy()


        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

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
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict


@register_dataset("swallow-rawvideo-rgb")
class SwallowRawVideoRGBDataset(SwallowDataset):
    '''
    Load raw video data from disk (only RGB frames)
    '''
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
        desired_actions=None, # desired action label names
        resize_to:int=None,  # resize the input features
        center_crop:bool=False, # center crop the input features
    ):
        super().__init__(
                is_training=is_training,     # if in training mode
                split=split,           # split, a tuple/list allowing concat of subsets
                feat_folder=feat_folder,     # folder for features
                json_file=json_file,       # json file for annotations
                feat_stride=feat_stride,     # temporal stride of the feats
                num_frames=num_frames,      # number of frames for each feat
                default_fps=default_fps,     # default fps
                downsample_rate=downsample_rate, # downsample rate for feats
                max_seq_len=max_seq_len,     # maximum sequence length during training
                trunc_thresh=trunc_thresh,    # threshold for truncate an action segment
                crop_ratio=crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
                input_dim=input_dim,       # input feat dim
                num_classes=num_classes,     # number of action categories
                file_prefix=file_prefix,     # feature file prefix if any
                file_ext=file_ext,        # feature file extension if any
                force_upsampling=force_upsampling, # force to upsample to max_seq_len
                feature_type=feature_type,
                two_stage=two_stage, # two stage training
                stage_at=stage_at,      # stage to start training
                desired_actions=desired_actions, # desired action label names
        )
        self.resize_to = resize_to
        self.center_crop = center_crop
    
    def centercrop_resize_video(self, feats, size=(128, 128)):
        if size[0] == 224:
            resized_len = 226
        elif size[0] == 128:
            resized_len = 130
        else:
            raise ValueError(f"Unsupported size {size}.")
        # feats: C x T x H x W
        # resize to large size first
        h, w = feats.shape[-2:]
        if h < resized_len or w < resized_len:
            d = resized_len - min(h, w)
            sc = 1 + d / min(h, w)
            feats = F.interpolate(feats, scale_factor=sc, mode='bilinear', align_corners=False)
        # center crop the input features
        i = int(np.round((h - size[0]) / 2.))
        j = int(np.round((w - size[1]) / 2.))
        feats = feats[:,:,i:i+size[0],j:j+size[1]]
        return feats

    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] +self.feature_type + self.file_ext)
        assert os.path.exists(filename), f"File {filename} does not exist"
        try:
            data = skvideo.io.vread(filename)
        except Exception as e:
            print(f"Error reading {filename}")
            raise e
        
        feats = torch.from_numpy(data).permute(3,0,1,2).float() # C x T x H x W

        if self.resize_to is not None and self.resize_to != feats.shape[-1]: # recommend to resize first
            if self.center_crop:
                feats = self.centercrop_resize_video(feats, size=(self.resize_to, self.resize_to))
            else:
                feats = F.interpolate(feats, size=(self.resize_to, self.resize_to), mode='bilinear', align_corners=False)

        # normalize the video
        feats = feats / 127.5 - 1.0
        # downsample the video
        feats = feats[:,::self.downsample_rate,...]
        feat_stride = self.feat_stride * self.downsample_rate

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


@register_dataset("swallow-rawvideo")
class SwallowRawVideoDataset(SwallowDataset):
    '''
    Load raw video data from disk (RGB and Flow)
    '''
    def __init__(self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for videos
        flow_folder,     # folder for flow features
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
        desired_actions=None, # desired action label names
        resize_to:int=None,  # resize the input features
    ):
        super().__init__(
                self,
                is_training,     # if in training mode
                split,           # split, a tuple/list allowing concat of subsets
                feat_folder,     # folder for videos
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
                feature_type,    # feature type (not used in this class)
                two_stage,       # two stage training
                stage_at,        # stage to start training
                desired_actions, # desired action label names
        )
        self.resize_to = resize_to
        self.flow_folder = flow_folder
        assert os.path.exists(self.flow_folder), f"Flow folder {self.flow_folder} does not exist"

    @staticmethod
    def get_flow_frames_from_targz(self, tar_dir):
        '''
        refer to https://ieeexplore.ieee.org/document/10244004/?arnumber=10244004
        '''
        list_u=[]
        list_v=[]
        with tarfile.open(tar_dir) as tar:
            mems=sorted(tar.getmembers(),key=lambda x:x.path)
            for x in mems:
                if(x.size==0):
                    continue
                filelikeobject=tar.extractfile(x)
                r=filelikeobject.read()
                bytes_stream = io.BytesIO(r)
                roiimg=Image.open(bytes_stream)
                nparr=np.array(roiimg,dtype=np.float)
                norm_data=nparr/127.5-1
                if(x.path.split("/")[1]=="u"):
                    list_u.append(torch.tensor(norm_data))
                else:
                    list_v.append(torch.tensor(norm_data))
        res_tensor=torch.stack([torch.stack(list_u),torch.stack(list_v)],dim=3)
        return res_tensor


    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] +self.feature_type + self.file_ext)
        assert os.path.exists(filename), f"File {filename} does not exist"
        try:
            data = skvideo.io.vread(filename)
        except Exception as e:
            print(f"RGB Video Error reading {filename}")
            raise e
        
        flow_filename = os.path.join(self.flow_folder, self.file_prefix + video_item['id'] +self.feature_type + '.tar.gz')
        assert os.path.exists(flow_filename), f"File {flow_filename} does not exist"
        try:
            flow_data = self.get_flow_frames_from_targz(flow_filename) # torch (T x H x W x 2)
        except Exception as e:
            print(f"Flow Video Error reading {flow_filename}")
            raise e
        flow_data = flow_data.permute(3,0,1,2).float() # 2 x T x H x W
        if self.resize_to is not None and self.resize_to != flow_data.shape[-1]:
            flow_data = F.interpolate(flow_data, size=(self.resize_to, self.resize_to), mode='bilinear', align_corners=False)

        feats = torch.from_numpy(data).permute(3,0,1,2).float() # C x T x H x W
        if self.resize_to is not None and self.resize_to != feats.shape[-1]:
            feats = F.interpolate(feats, size=(self.resize_to, self.resize_to), mode='bilinear', align_corners=False)

        # normalize the video
        feats = feats / 255.0
        # concat the RGB and Flow features on the channel dimension
        feats = torch.cat([feats, flow_data], dim=0)

        # downsample the video
        feats = feats[:,::self.downsample_rate,...]
        feat_stride = self.feat_stride * self.downsample_rate

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

@register_dataset("two-modal")
class MultiModalDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(dataset1) == len(dataset2), "video list should be the same"
        
    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]