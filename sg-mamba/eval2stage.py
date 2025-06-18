# python imports
import argparse
from functools import partial
import pickle
import os, tarfile, io
import glob
import time
import json
from collections import OrderedDict
from pprint import pprint
from copy import deepcopy

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import infer_one_epoch, ANETdetection, fix_random_seed,\
                    valid_one_epoch, crop_video, VideoKeypointProcessor, slideTensor


from pytorch_i3d import InceptionI3d

# thir party imports
import skvideo.io
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def run(cfg, args, action_label=None, infer_or_eval=infer_one_epoch, eval_label_dict=None):
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(
            glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)
    """2. create dataset / dataloader"""
    if getattr(args, 'train_set', False):
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['train_split'],
                                   **cfg['dataset'])
    else:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
    # set bs = 1, and disable shuffle
    cfg['loader']['batch_size'] = 1
    val_loader = make_data_loader(val_dataset, False, None, **cfg['loader'])
    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
    # load ema model instead
    print("Loading from EMA model ...")
    if getattr(args, 'resnet_ateval', False):
        from torchvision.models.resnet import ResNet50_Weights
        weights = OrderedDict()
        old_w = checkpoint['state_dict_ema']
        for k, v in old_w.items():
            # replace head
            if k!='module.image_embed.proj.weight' and k!='module.image_embed.proj.bias':
                weights[k] = v
            if k == 'module.image_embed.proj.weight':
                weights['module.image_embed.module.fc.weight'] = v
            if k == 'module.image_embed.proj.bias':
                weights['module.image_embed.module.fc.bias'] = v
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        # assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    else:
        model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            only_focus_on=cfg['dataset']['desired_actions'] if args.desired_actions is None else args.desired_actions,
            single_action= args.eval_single_cls,
            )
    else:
        output_file = os.path.join(
            os.path.split(ckpt_file)[0], 'eval_results.pkl')
    """5. Test the model"""
    print("\nStart inferring model {:s} ...".format(cfg['model_name']))
    start = time.time()
    train_label_dict = val_dataset.label_dict
    result = infer_or_eval(val_loader,
                          model,
                          -1,
                          ext_score_file=cfg['test_cfg']['ext_score_file'],
                          evaluator=det_eval,
                          output_file=output_file,
                          pflops=args.calflops,
                        #   visualize=args.visualize,
                        #   print_freq=args.print_freq,
                          )
    
    result = remap_action_labels(result, train_label_dict, eval_label_dict)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))
    return result

def remap_action_labels(result, train_label_dict, eval_label_dict):
    # check if we need to remap
    remap = False
    if eval_label_dict is not None:
        for label, id in train_label_dict.items():
            if label not in eval_label_dict:
                print(f"Warning: label {label} not in eval_label_dict.")
            if train_label_dict[label] != eval_label_dict[label]:
                remap = True
                break
     # remap action labels
    if remap:
        for label, train_label_id in train_label_dict.items():
            if label in eval_label_dict:
                result['label'][result['label'] == train_label_id] = eval_label_dict[label] + 1000
            else:
                print(f"Warning: {label} not found in eval_label_dict")
        result['label'] -= 1000
    return result

def get_label_dict(json_path, desired_actions):
    label_dict = {}
    with open(json_path, 'r') as f:
        json_db = json.load(f)

    for key, value in json_db.items():
        for act in value['annotations']:
            if desired_actions is not None and act['label'] in desired_actions:
                label_dict[act['label']] = act['label_id']
            elif desired_actions is None:
                label_dict[act['label']] = act['label_id']
            elif act['label'] not in desired_actions:
                continue
    return label_dict

def merge_keypoint_config(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    if os.path.isfile(args.config2):
        cfg2 = load_config(args.config2)
        cfg['keypoint'] = cfg2['keypoint']
    return cfg
        
    
################################################################################
def main(args):
    ### Stage 1   
    cfg, eval_dataset, eval_db_vars, new_feat_path, new_feat_center, new_json_path = stage1infer_extractFeature(args)

    ### Stage 2
    if args.config2 and not args.only_perfect:
        stage2eval(args, eval_dataset, eval_db_vars, new_feat_center, new_feat_path, new_json_path)
    
    if args.config2 and args.infer_perfect_stage1:
        # if cache exists, load it
        save_cache_name = os.path.basename(args.ckpt).split(".pth.tar")[0] + "_perfect.pkl"
        save_cache_path = os.path.join(args.perfect_stage1, save_cache_name)
        cfg['heatmap_type'] = args.heatmap_type
        cfg['heatmap_dir'] = args.heatmap_dir
        cfg['image_size'] = args.image_size
        cfg['heatmap_size'] = args.heatmap_size
        cfg['heatmap_branch'] = args.heatmap_branch
        cfg['heatmap'] = args.heatmap
        cfg['kalman'] = args.kalman
        cfg['normal_kalman'] = args.normal_kalman
        cfg['keypoint']['sigma'] = args.heatmap_sigma if args.heatmap else cfg['keypoint']['sigma']
        cfg['selected_index'] = args.selected_index
        if os.path.isfile(save_cache_path):
            print(f"Loading cache from {save_cache_path}")
            with open(save_cache_path, 'rb') as f:
                cache = pickle.load(f)
            perfect_result = cache['result']
            perfect_feat_center = cache['new_feat_center']
            perfect_json_path = cache['new_json_path']
        else:
            # re-extract features
            # do not support crop features
            # extract features for perfect stage 1
            cfg['cache_dir'] = args.perfect_stage1
            os.makedirs(args.perfect_stage1, exist_ok=True)
            perfect_result = build_perfect_stage1_results(eval_dataset.json_file, cfg['val_split'], args.seg_duration)
            if args.heatmap_branch in ['rgb', 'flow'] or not args.heatmap:
                initI3ds(args)
            perfect_feat_center = extract_features_from_res(args.video_root, args.perfect_stage1, args.flow_dir, perfect_result, cfg)
            perfect_json_path = build_tmp_json(cfg, perfect_feat_center)
            cfg['cache_dir'] = args.cache_dir

            # save cache
            cache = {}
            cache['result'] = perfect_result
            cache['new_feat_center'] = perfect_feat_center
            cache['new_json_path'] = perfect_json_path
            with open(save_cache_path, 'wb') as f:
                pickle.dump(cache, f)
        
        perfect_json_path = build_tmp_json(cfg, perfect_feat_center)
        print("Evaluating based on perfect stage 1 ...")
        stage2eval(args, eval_dataset, eval_db_vars, perfect_feat_center, args.perfect_stage1, perfect_json_path)

def stage1infer_extractFeature(args):
    """0. load config"""
    # sanity check
    cfg = merge_keypoint_config(args)
    cfg['cache_dir'] = args.cache_dir
    cfg['cropped_videos'] = args.cropped_videos
    cfg['heatmap_dir'] = args.heatmap_dir
    cfg['image_size'] = args.image_size
    cfg['heatmap_size'] = args.heatmap_size
    cfg['heatmap_branch'] = args.heatmap_branch
    cfg['heatmap'] = args.heatmap
    cfg['keypoint']['sigma'] = args.heatmap_sigma
    cfg['seg_duration'] = args.seg_duration
    cfg['heatmap_type'] = args.heatmap_type
    cfg['kalman'] = args.kalman
    cfg['normal_kalman'] = args.normal_kalman
    cfg['selected_index'] = args.selected_index
    pprint(cfg)
    args.saveonly = True
    if args.train_set:
        eval_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['train_split'], **cfg['dataset']
        )
    else:
        eval_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
    eval_db_vars = eval_dataset.get_attributes()
    # get action id dict from json file

    if args.test_first_stage:
        det_eval_stage1 = ANETdetection(
            eval_dataset.json_file,
            eval_dataset.split[0],
            tiou_thresholds=eval_db_vars['tiou_thresholds'],
            only_focus_on=['AllTime'],
        )
    
    # init
    new_feat_path = args.cache_dir
    os.makedirs(new_feat_path, exist_ok=True)

    # if cache exists, load it
    save_cache_name = os.path.basename(args.ckpt).split(".pth.tar")[0] + ".pkl"
    save_cache_path = os.path.join(args.cache_dir, save_cache_name)
    print(f"save_cache_path: {save_cache_path}")
    if os.path.isfile(save_cache_path):
        print(f"Loading cache from {save_cache_path}")
        with open(save_cache_path, 'rb') as f:
            cache = pickle.load(f)
        result = cache['result']
        new_feat_center = cache['new_feat_center']
        new_feat_path = args.cache_dir
        new_json_path = cache['new_json_path']
        # if data not extracted, re-extract
        feat_num = len([i for i in os.listdir(new_feat_path) if i.endswith('.npy')])
        if feat_num != len(new_feat_center):
            if args.re_extract:
                if args.heatmap_branch in ['rgb', 'flow'] or not args.heatmap:
                    initI3ds(args)
                feat_center = extract_features_from_res(args.video_root, new_feat_path, args.flow_dir, result, cfg)
                # assert feat_center == new_feat_center, "Re-extracted features are not the same as the original ones."
    else:
        result = run(cfg, args, action_label=cfg['dataset']['desired_actions'], infer_or_eval=infer_one_epoch)
        # {video-id:[], t-start:[], t-end:[], score:[], label:[]}
        result['t-center'] = (result['t-start'] + result['t-end']) / 2

        if args.test_first_stage:
            det_eval_stage1.evaluate(result)

        print("Before filtering, we have ", len(result['video-id']), "segments.")

        # filter out low confidence results
        mask = result['score'] > args.confidence
        
        new_ids = []
        for key in result.keys():
            if key == 'video-id':
                for idx in range(len(result[key])):
                    if mask[idx]:
                        new_ids.append(result[key][idx])
                result[key] = new_ids
            else:
                result[key] = result[key][mask]
        
        print("After filtering, we have ", len(result['video-id']), "segments.")

        if len(set(result['video-id'])) < len(eval_dataset):
            print(f"Warning: length of stage 1 is smaller than the dataset, \
            {len(set(result['video-id']))} vs {len(eval_dataset)}. Try to decrease confidence threshold.")

        # re-extract features
        if args.re_extract:
            # get center and extend 
            video_root = args.video_root
            assert os.path.isdir(video_root), "Video root does not exist!"
            if args.heatmap_branch in ['rgb', 'flow'] or not args.heatmap:
                print("====Loading I3D Model=====")
                initI3ds(args)
            # extract features
            new_feat_center = extract_features_from_res(video_root, new_feat_path, args.flow_dir, result, cfg)
        else: # much faster but worser performance (drop 10% mAP)
            new_feat_center = crop_features_from_res(cfg, new_feat_path, result)
            
        # build new json file for stage 2 dataset
        new_json_path = build_tmp_json(cfg, new_feat_center)

        # save pickle 
        cache = {
            'result': result,
            'new_feat_center': new_feat_center,
            'new_json_path': new_json_path
        }
        with open(save_cache_path, 'wb') as f:
            pickle.dump(cache, f)
    
    if args.test_first_stage:
        # after filtering, we have a new result
        print('filtering results with threshold: ', args.confidence) 
        mAP = det_eval_stage1.evaluate(result)
    return cfg,eval_dataset,eval_db_vars,new_feat_path,new_feat_center,new_json_path


def crop_features_from_res(cfg, new_feat_path, result):
    print("Warning: Clipping features rather than re-extracting from videos, this will lead to worser performance.")
    CLIP_DUR = cfg['seg_duration']
            # get the original feature
    old_feat_root = cfg['dataset']['feat_folder']
    video_rank, cur_video_id = 0, None
    new_feat_center = {}
    for idx, res in tqdm(enumerate(result['video-id'])):
        feat_path = os.path.join(old_feat_root, f"{res}.npy")
        assert os.path.isfile(feat_path), "Feature file does not exist!"
        if cur_video_id != res:
            cur_video_id = res
            video_rank = 0
        video_rank += 1
        seg_id = f'{res}#{video_rank}'
        crop_feat_path = os.path.join(new_feat_path, f"{seg_id}.npy")
        data = np.load(feat_path)
        duration = int(res.split("_")[-1].split("s")[0])
        clip_start = result['t-center'][idx] - CLIP_DUR / 2
        clip_end = result['t-center'][idx] + CLIP_DUR / 2
        clip_start = max(clip_start, 0)
        clip_end = min(clip_end, duration)
        start_ratio = clip_start / duration
        end_ratio = clip_end / duration
        start_idx = int(start_ratio * data.shape[0])
        end_idx = int(end_ratio * data.shape[0])
        new_data = data[start_idx:end_idx]
        np.save(crop_feat_path, new_data)
        new_feat_center[seg_id] = result['t-center'][idx]
    return new_feat_center

def stage2eval(args, eval_dataset, eval_db_vars, new_feat_center, new_feat_path, new_json_path):
    results = {
            'seg-id': [],
            't-start': [],
            't-end': [],
            'score': [],
            'label': [],
            'video-id': []
    }
    if os.path.isfile(args.config2):
        cfg = load_config(args.config2)
    else:
        raise ValueError("Config file does not exist.")
    cfg['raise_error'] = args.raise_error
    args.saveonly = False
    cfg['dataset']['feat_folder'] = new_feat_path
    cfg['dataset']['json_file'] = new_json_path
    cfg['val_split'] = ['test']
    if args.desired_actions is not None:
        cfg['dataset']['desired_actions'] = args.desired_actions
    pprint(cfg)
    det_eval = ANETdetection(
        eval_dataset.json_file,
        eval_dataset.split[0],
        tiou_thresholds=eval_db_vars['tiou_thresholds'],
        only_focus_on=cfg['dataset']['desired_actions'],
        single_action=args.eval_single_cls,
    )
    # get action id dict from json file
    label_dict = get_label_dict(eval_dataset.json_file, cfg['dataset']['desired_actions'])
    if cfg['dataset']['num_classes'] == 1:  # single classification
        for action in cfg['dataset']['desired_actions']:
            action_id = single_cls_map(args, cfg, label_dict, action)
            result = run(cfg, args, action_label=action, infer_or_eval=infer_one_epoch)

            # gather results (numpy) from different actions
            results['seg-id'].extend(result['video-id'])
            results['t-start'].extend(result['t-start'].tolist())
            results['t-end'].extend(result['t-end'].tolist())
            results['score'].extend(result['score'].tolist())
            results['label'].extend([action_id]* len(result['score']))
            results['video-id'].extend([i.split("#")[0] for i in result['video-id']])
            

        # transform
        results['t-start'] = torch.tensor(results['t-start']).numpy()
        results['t-end'] = torch.tensor(results['t-end']).numpy()
        results['label'] = torch.tensor(results['label']).numpy()
        results['score'] = torch.tensor(results['score']).numpy()
    else:                                   # multiple classification
        args.ckpt = args.ckpt2
        results = run(cfg, args, action_label=cfg['dataset']['desired_actions'], infer_or_eval=infer_one_epoch, eval_label_dict=label_dict)
        results['seg-id'] = results['video-id']
        results['video-id'] = [i.split("#")[0] for i in results['video-id']]
    # shift t-start and t-end based on stage 1 segment results (+ t-center - 2)
    results = shift_result(new_feat_center, results, args.seg_duration)
    if args.result_path:
        dump_result_path = os.path.join(args.result_path, 'final_result.pkl')
        with open(dump_result_path, 'wb') as f:
            pickle.dump(results, f)
    # evaluate
    mAP = det_eval.evaluate(results) # should be evaluated on the original video rather than the clipped video
    print(f"mAP: {mAP[0] * 100}")
    if args.eval_single_cls:
        activity_index = mAP[-1]
        ap = mAP[-2]
        for action, i in activity_index.items():
            mean_ap = ap[:, i].mean()
            print(f"Action: {action}, AP: {ap[:, i]}, Mean AP: {mean_ap}")

def single_cls_map(args, cfg, label_dict, action):
    action_id = label_dict[action]
    # search action ckpt dir
    action_ckpt_dirs = os.listdir(args.ckpt2)
    action_ckpt_dirs = [ckpt_dir for ckpt_dir in action_ckpt_dirs if action in ckpt_dir]
    if len(action_ckpt_dirs) > 1:
        print(f"Warning: Multiple ckpt dirs found for action {action}, using the first one.")
    elif len(action_ckpt_dirs) == 0:
        print(f"Error: No ckpt dir found for action {action}.")
        raise FileNotFoundError
    action_ckpt_dir = os.path.join(args.ckpt2, action_ckpt_dirs[0])
    if args.last_epoch or args.epoch:
        args.ckpt = get_best_pth_from_dir(action_ckpt_dir, 'epoch', args.epoch)
    else:
        args.ckpt = get_best_pth_from_dir(action_ckpt_dir)
    print(f"Using ckpt: {args.ckpt}")
    cfg['dataset']['desired_actions'] = [action]
    return action_id

def shift_result(new_feat_center, results, seg_duration):
    shift = seg_duration / 2
    for idx in range(len(results['video-id'])):
        # get seg_id
        seg_id = results['seg-id'][idx]
        center = new_feat_center[seg_id]
        # shift
        results['t-start'][idx] += center - shift
        results['t-end'][idx] += center - shift
    return results


def build_perfect_stage1_results(json_file, split='test', seg_duration=4.004):
    # {video-id:[], t-start:[], t-end:[], score:[], label:[]}
    with open(json_file, 'r') as f:
        gt = json.load(f)
    all_time_seg = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'score': [],
        'label': [],
        't-center': []
    }
    if isinstance(split, str):
        split = [split.lower()]
    split = [s.lower() for s in split]
    for video_name, video_info in gt.items():
        if  video_info['subset'].lower() not in split:
            continue
        rank = 0
        anno = video_info['annotations']
        for action in anno:
            if action['label'] != 'AllTime':
                continue
            start = action['segment'][0]
            end = action['segment'][1]
            center = (start + end) / 2
            new_start = max(0, center - seg_duration / 2)
            new_end = min(video_info['duration'], center + seg_duration / 2)
            rank += 1
            all_time_seg['video-id'].append(video_name)
            all_time_seg['t-start'].append(new_start)
            all_time_seg['t-end'].append(new_end)
            all_time_seg['score'].append(1)
            all_time_seg['label'].append(0)
            all_time_seg['t-center'].append(center)
    return all_time_seg

def extract_features_from_res(video_root, new_feat_path, flow_dir, result, cfg):
    # define some constants
    IMAGE_SIZE, HEATMAP_SIZE = cfg['image_size'], cfg['heatmap_size']
    CLIP_DUR = cfg['seg_duration']
    WINDOW_SIZE = cfg['dataset']['num_frames']
    WINDOW_STEP = cfg['dataset']['feat_stride']
    kalman = True if cfg['kalman'].lower() == 'true' else False
    normal_kalman = cfg['normal_kalman']
    # get center and extend 
    cache_video_id = None
    rgb_data = None
    flow_data = None
    new_feats = {}
    per_video_rank = 0
    processor = None
    if cfg['heatmap']:
        from libs.utils import VideoKeypointProcessor2
        print(f"Loading Ckpt from {cfg['keypoint']['model_path']}")
        processor = VideoKeypointProcessor2(cfg['keypoint']['model_path'], sigma=cfg['keypoint']['sigma'])

    for idx in tqdm(range(len(result['video-id']))):
        video_id = result['video-id'][idx]
        t_center = result['t-center'][idx]
        t_extend = CLIP_DUR # 4 seconds
        clip_start = t_center - t_extend / 2
        clip_end = t_center + t_extend / 2
        duration = int(video_id.split("_")[-1].split("s")[0])
        clip_start = max(clip_start, 0)
        clip_end = min(clip_end, duration)
        start_ratio = clip_start / duration
        end_ratio = clip_end / duration
        
        if not cfg['cropped_videos']:
            if cache_video_id != video_id:
                cache_video_id = video_id
                per_video_rank = 0
                video_path = os.path.join(video_root, f"{video_id}.avi")
                assert os.path.isfile(video_path), f"Video file {video_path} does not exist!"
                try:
                    rgb_data = skvideo.io.vread(video_path)
                except Exception as e:
                    print(f"Error reading video: {video_path}")
                    if cfg['raise_error']:
                        raise e
                    else:
                        print(e)
                    continue
                if not cfg['heatmap']:
                    # resize video
                    rgb_data=torch.from_numpy(rgb_data) 
                    if rgb_data.shape[1]!=IMAGE_SIZE or rgb_data.shape[2]!=IMAGE_SIZE:
                        rgb_data = resize_data(rgb_data, IMAGE_SIZE)
                    rgb_data=rgb_data.view(-1,IMAGE_SIZE,IMAGE_SIZE,3) / 127.5 - 1
                    rgb_data=rgb_data.float()
                    preprocess = None
                # extract keypoint
                else:
                    preprocess = partial(extract_keypoints, processor=processor, 
                    HEATMAP_SIZE=HEATMAP_SIZE, branch=cfg['heatmap_branch'], heatmap_type=cfg['heatmap_type'])
                
                if flow_dir is not None and not cfg['heatmap']:
                    flow_data=get_flow_frames_from_targz(os.path.join(flow_dir, f"{video_id}.tar.gz"))
                    if flow_data.shape[1]!=IMAGE_SIZE or flow_data.shape[2]!=IMAGE_SIZE:
                        flow_data = resize_data(flow_data, IMAGE_SIZE)
                    flow_data=flow_data.view(-1,IMAGE_SIZE,IMAGE_SIZE,2).float()
            else:
                per_video_rank += 1
        
        else:#NOTE: cropped videos only support heatmap, which is ok for now
            if cache_video_id != video_id:
                cache_video_id = video_id
                per_video_rank = 0
            else:
                per_video_rank += 1
            heatmap_path = os.path.join(cfg['heatmap_dir'], 'heatmap', f"{video_id}#{per_video_rank}.npy")
            if os.path.exists(heatmap_path):    # already extracted heatmap, read from cache
                rgb_data = np.load(heatmap_path)
            else: # extract heatmap from video
                # crop video first
                video_path = os.path.join(video_root, f"{video_id}.avi")
                output_path = os.path.join(cfg['heatmap_dir'], 'cropped_video',f"{video_id}#{per_video_rank}.avi")
                
                if not os.path.exists(output_path):
                    crop_video(video_path, output_path, clip_start, clip_end)

                # extract heatmap
                if processor is None:
                    print(f"Loading Ckpt from {cfg['keypoint']['model_path']}")
                    processor = VideoKeypointProcessor(cfg['keypoint']['model_path'], sigma=cfg['keypoint']['sigma'])
                cropped_fusion = processor.infer_heatmaps(output_path, kalman=kalman, normal_kalman=normal_kalman)[cfg['selected_index']]
                # save heatmap
                os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                np.save(heatmap_path, cropped_fusion)
                # get heatmap
                rgb_data = cropped_fusion

            rgb_data = torch.from_numpy(rgb_data).unsqueeze(1)
            rgb_data = torch.nn.functional.interpolate(rgb_data, (HEATMAP_SIZE, HEATMAP_SIZE), mode='bilinear', align_corners=False)
            rgb_data = rgb_data.repeat(1, 3, 1, 1)
            rgb_data = rgb_data.permute(0, 2, 3, 1).double()

        # slidedata_flow=slideTensor(flow_data,args.chunk_size,args.frequency)
        saved_new_feat_name = video_id + f"#{per_video_rank}"
        new_feat_file = os.path.join(new_feat_path, f"{saved_new_feat_name}.npy")
        new_feats[saved_new_feat_name] = t_center
        if os.path.isfile(new_feat_file):
            continue
        # extract features
        if cfg['heatmap']:
            flow_data = None
            extract_features(rgb_data, flow_data, new_feat_file, 
                start_ratio, end_ratio, WINDOW_SIZE, WINDOW_STEP, preprocess=preprocess, cropped=cfg['cropped_videos'], branch=cfg['heatmap_branch'])
        else:
            extract_features(rgb_data, flow_data, new_feat_file, 
                start_ratio, end_ratio, WINDOW_SIZE, WINDOW_STEP, preprocess=preprocess, cropped=cfg['cropped_videos'])
    return new_feats

def extract_keypoints(video_data, processor, HEATMAP_SIZE, branch, heatmap_type='fusion'):
    '''
        Please make sure use this function when the video_data is cropped rather than the whole video,
        otherwise it will lead to worser performance.
    '''
    if heatmap_type.lower() == 'fusion':
        index = -1
    elif heatmap_type.lower() == 'keypoint':
        index = 0
    elif heatmap_type.lower() == 'line':
        index = 1
    else:
        raise ValueError("Invalid heatmap type")
    fusion_data = processor.infer_heatmaps(video_data)[index]
    rgb_data = torch.from_numpy(fusion_data).unsqueeze(1)
    rgb_data = torch.nn.functional.interpolate(rgb_data, (HEATMAP_SIZE, HEATMAP_SIZE), mode='bilinear', align_corners=False)
    if branch == 'rgb':
        rgb_data = rgb_data.repeat(1, 3, 1, 1)
    elif branch == 'flow':
        rgb_data = rgb_data.repeat(1, 2, 1, 1)
    elif branch == 'none' or branch == '' or branch is None:
        pass
    else:
        raise ValueError("Invalid branch value")
    rgb_data = rgb_data.permute(0, 2, 3, 1).double()
    return rgb_data

def get_flow_frames_from_targz(tar_dir):
    '''ref to swallow_a2net_vswg/tools/eval.py'''
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
            nparr=np.array(roiimg,dtype=np.float64)
            norm_data=nparr/127.5-1
            if(x.path.split("/")[1]=="u"):
                list_u.append(torch.tensor(norm_data))
            else:
                list_v.append(torch.tensor(norm_data))
    res_tensor=torch.stack([torch.stack(list_u),torch.stack(list_v)],dim=3)
    return res_tensor

def extract_features(rgb_data, flow_data, new_feat_file, start_ratio, end_ratio, win_size, win_step, preprocess=None, cropped=False, branch='rgb'):
    rgb_time_long = rgb_data.shape[0]
    mode = 'rgb'
    # clip
    if not cropped:
        rgb_start_idx = max(0, int(start_ratio * rgb_time_long))
        rgb_end_idx = min(rgb_time_long, int(end_ratio * rgb_time_long))
        rgb_data = rgb_data[rgb_start_idx:rgb_end_idx]
    if preprocess is not None:
        rgb_data = preprocess(rgb_data)
        mode = branch

    if branch is None or branch == '' or branch.lower() == 'none':
        # do not extract but save only
        np.save(new_feat_file, rgb_data.squeeze(-1))
        return rgb_data
    # slide
    rgb_data = slideTensor(rgb_data, win_size, win_step)
    # get feat
    feat_spa=get_features(rgb_data, mode, batch_size=-1)
    feat_spa=torch.from_numpy(feat_spa)

    if flow_data is not None:
        flow_time_long = flow_data.shape[0]
        flow_start_idx = max(0, int(start_ratio * flow_time_long))
        flow_end_idx = min(flow_time_long, int(end_ratio * flow_time_long))
        flow_data = flow_data[flow_start_idx:flow_end_idx]
        flow_data = slideTensor(flow_data, win_size, win_step)
        feat_tem=get_features(flow_data,"flow", batch_size=-1)
        feat_tem=torch.from_numpy(feat_tem)
        # concat rgb and flow features
        feat = np.concatenate([feat_spa, feat_tem], axis=1)
    else:
        feat = feat_spa
    # save feat
    np.save(new_feat_file, feat)
    return feat

def get_features(data, mode, batch_size=32):
    '''
    data: (T, 8, W, H, C)
    mode: 'rgb' or 'flow'
    return data with shape: (T, 1024)
    '''
    data = data.permute(0, 4, 1, 2, 3) # (T, C, 8, W, H)
    if batch_size == -1:
        batch_size = data.shape[0]
    batch_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    if mode == 'rgb':
        i3d = i3d_rgb
    else:
        i3d = i3d_flow
    
    all_features = []
    for batch in batch_loader:
        batch = batch.float().cuda()
        with torch.no_grad():
            features = i3d.extract_features(batch)
        # output: (batch_size, 1024, 1, 1, 1)
        features = features[:, :, 0, 0, 0].cpu().numpy()
        all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)
    return all_features

def build_tmp_json(cfg, new_feat_center):
    old_json = cfg['dataset']['json_file']
    with open(old_json, 'r') as f:
        data = json.load(f)
    CLIP_DUR = cfg['seg_duration']
    new_data = {}
    for seg_id, t_center in new_feat_center.items():
        shift = t_center - CLIP_DUR / 2
        video_id = seg_id.split("#")[0]
        new_data[seg_id] = deepcopy(data[video_id])
        new_data[seg_id]['duration'] = CLIP_DUR
        new_data[seg_id]['annotations'] = []
        new_data[seg_id]['shift'] = shift

        for anno in data[video_id]['annotations']:
            new_anno = deepcopy(anno)
            new_anno['segment'][0] -= shift
            new_anno['segment'][1] -= shift

            if new_anno['segment'][0] < 0 or new_anno['segment'][1] < 0:
                continue
            if new_anno['segment'][0] > CLIP_DUR or new_anno['segment'][1] > CLIP_DUR:
                continue

            new_anno.pop("segment(frames)")
            new_data[seg_id]['annotations'].append(new_anno)
    new_json_path = os.path.join(cfg['cache_dir'], 'tmp.json')
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    return new_json_path

def initI3ds(args):
    global i3d_rgb,i3d_flow
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load(args.flow_i3d))
    i3d_rgb = InceptionI3d(7, in_channels=3)
    new_kv=OrderedDict()
    old_kv=torch.load(args.rgb_i3d)['state_dict']
    for k,v in old_kv.items():
        new_kv[k.replace("module.","")]=v
    i3d_rgb.load_state_dict(new_kv)
    #i3d_rgb = InceptionI3d(400, in_channels=3)
    #i3d_rgb.load_state_dict(torch.load(load_model_rgb))
    i3d_rgb.train(False)
    i3d_flow.train(False)
    i3d_rgb.cuda()
    i3d_flow.cuda()
    print(f'load i3d_rgb and i3d_flow')

def resize_data(data, image_size):
    data_tmp=torch.zeros(data.shape[0:1]+(image_size,image_size,data.shape[-1])).float()
    for index, datum in enumerate(data):
        datum_tmp = torch.from_numpy(cv2.resize(datum.numpy(),(image_size,image_size))).float()
        data_tmp[index,:,:,:] = datum_tmp
    return data_tmp

def get_best_pth_from_dir(dir,key='performance',epoch=-1) -> str:
    assert os.path.isdir(dir), "Directory does not exist!"
    ckpts = os.listdir(dir)
    ckpts = [ckpt for ckpt in ckpts if ".pth.tar" in ckpt]
    if key == 'performance':
        ckpts = sorted(ckpts, key=lambda x: float(x.split(".pth.tar")[0].split("_")[-1]), reverse=True)
    elif key == 'epoch':
        ckpts = sorted(ckpts, key=lambda x: int(x.split(".pth.tar")[0].split("_")[-2]), reverse=True)
        if epoch != -1:
            ckpts = [ckpt for ckpt in ckpts if f'epoch_{epoch:03d}' in ckpt]
    return os.path.join(dir, ckpts[0])

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument(
        '--config',
        type=str,
        metavar='DIR',
        default='configs/2stage/mamba_swallow_i3d_eval_stage1.yaml',
        help='path to a config file for stage 1')
    parser.add_argument('--config2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to a config file for stage 2')
    parser.add_argument('--ckpt2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to checkpoints for stage 2')
    parser.add_argument('--ckpt',
                        type=str,
                        metavar='DIR',
                        default='ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar',
                        help='path to a checkpoint')
    parser.add_argument('--epoch',
                        type=int,
                        default=None,
                        help='checkpoint epoch')
    parser.add_argument('-t',
                        '--topk',
                        default=-1,
                        type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument(
        '--saveonly',
        action='store_true',
        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument("-v",
                        "--visualize", 
                        action='store_true',
                        help="visualize the results")
    parser.add_argument("--train_set", action='store_true', help="eval train set")
    parser.add_argument('--confidence', type=float, default=0.23, help='confidence threshold for stage 1 results')
    parser.add_argument("--re-extract", action='store_true', help="whether to re-extract features at stage 2")
    parser.add_argument('--video_root', type=str, metavar='DIR', help='path to video root (specific for stage 2)')
    parser.add_argument('--cache_dir', type=str, metavar='DIR', default='./cache', help='path to cache dir for the extracted features (specific for stage 2)')
    parser.add_argument('--flow_dir', type=str, metavar='DIR',  help='path to flow dir (specific when re-extract)')
    parser.add_argument("--flow_i3d", type=str, metavar='DIR', default='./pretrained/flow_imagenet.pt', help='path to flow i3d model')
    parser.add_argument("--rgb_i3d", type=str, metavar='DIR', default='./pretrained/pretrained_swallow_i3d.pth', help='path to rgb i3d model')
    parser.add_argument("--raise_error", action='store_true', help="raise error when video reading error")
    parser.add_argument("--test_first_stage", action='store_true', help="test first stage on AllTime")
    parser.add_argument("--heatmap", action='store_true', help="use heatmap as input") #TODO: better remove it with re-implementation
    parser.add_argument("--cropped_videos", action='store_true', help="use cropped videos instead of reading from whole vidoe")
    parser.add_argument("--heatmap_dir", type=str, metavar='DIR', default='tmp/heatmaps', help='desired save path to heatmap dir, use it when cropped_videos is True')
    parser.add_argument("--image_size", type=int, default=128, help='image size for heatmap')
    parser.add_argument("--heatmap_size", type=int, default=224, help='heatmap size for heatmap')
    parser.add_argument("--heatmap_branch", type=str, default='rgb', choices=['rgb', 'flow', 'none'], help='i3d branch for heatmap')
    parser.add_argument("--heatmap_sigma", type=float, default=4.0, help='sigma for heatmap')
    parser.add_argument("--resnet_ateval", action='store_true', help="probe resnet at eval (extracted resnet feature at train)")
    parser.add_argument("--infer_perfect_stage1", action='store_true', help="infer on best stage 1")
    parser.add_argument("--perfect_stage1", type=str, metavar='DIR', default='', help='path to extracted features')
    parser.add_argument("--seg_duration", type=float, default=4.004, help='segment duration for stage 2')
    parser.add_argument("--result_path", type=str, default=None, help='output result path')
    parser.add_argument("--last_epoch", action='store_true', help='use last epoch to evaluate(default: best epoch)')
    parser.add_argument("--only_perfect", action='store_true', help='only evaluate result based on perfect stage 1')
    parser.add_argument("--heatmap_type", type=str, default='fusion', choices=['fusion', 'keypoint', 'line'], help='heatmap type')
    parser.add_argument('--kalman', choices=['true', 'false','True','False'], default='true', help='use kalman to extract heatmaps')
    parser.add_argument('--normal_kalman', action='store_true', default=False)
    parser.add_argument('--selected_index', type=int, default=-1)
    parser.add_argument('--calflops', action='store_true', help='calculate flops')
    parser.add_argument('--desired_actions', nargs='+', default=None)
    parser.add_argument('--eval_single_cls', action='store_true', help='evaluate single class action')
    args = parser.parse_args()
    main(args)

