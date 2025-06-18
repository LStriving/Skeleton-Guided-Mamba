# python imports
import argparse
import os
import pickle
from pprint import pprint
import glob

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.datasets.swallow import MultiModalDataset
from libs.modeling import make_meta_arch, make_two_tower
from libs.utils import (infer_one_epoch, ANETdetection,
                        fix_random_seed)
from train2stage import get_label_dict_from_file
from eval2stage import (get_label_dict, remap_action_labels, shift_result, single_cls_map, stage1infer_extractFeature)

def get_cfg(config_file):
    if os.path.isfile(config_file):
        cfg = load_config(config_file)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    return cfg

def run2tower(cfg, cfg2, args, action_label=None):
    # get checkpoint file
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpts = os.listdir(args.ckpt)
            ckpt_name = [i for i in ckpts if 'epoch_{:03d}'.format(args.epoch) in i]
            assert len(ckpt_name) != 0, f"Epoch {args.epoch} for ckpt not found"
            ckpt_name = ckpt_name[0]
            ckpt_file = os.path.join(args.ckpt, ckpt_name)
        else:
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # set batch size to 1 for evaluation
    cfg['loader']['batch_size'] = 1
    cfg2['loader']['batch_size'] = 1
    if args.train_set:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['train_split'],
                                   **cfg['dataset'])
        val_dataset2 = make_dataset(cfg2['dataset_name'], False, cfg2['train_split'],
                                   **cfg2['dataset'])
    else:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
        val_dataset2 = make_dataset(cfg2['dataset_name'], False, cfg2['val_split'],
                                **cfg2['dataset'])
    multi_valdataset = MultiModalDataset(val_dataset, val_dataset2) 
    cfg['loader']['batch_size'] = 1
    val_loader = make_data_loader(multi_valdataset, False , rng_generator, **cfg['loader'])
    if args.vw is not None:
        cfg['two_tower']['vw'] = args.vw
        print(f"Replace confusion weight factor vw to {args.vw}")
    
    # load model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])
    # two-tower model
    model = make_two_tower(args.tower_name, model, model2, cfg, cfg2, **cfg['two_tower'])
    # not ideal for multi GPU training, ok for now
    model_eval = nn.DataParallel(model, device_ids=cfg['devices'])

    """load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
    # load ema model instead
    print("Loading from EMA model ...")
    model_eval.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """5. Test the model"""
    print(f"\nStart inferring model {cfg['model_name']} and {cfg2['model_name']} ...")

    
    train_label_dict = val_dataset.label_dict
    eval_label_dict = get_label_dict_from_file(cfg['dataset']['json_file'], action_label)

    
    result = infer_one_epoch(
        val_loader,
        model_eval,
        -1,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=20, #args.print_freq
        pflops=args.calflops
    )
    # remap action labels
    result = remap_action_labels(result, train_label_dict, eval_label_dict)
    return result

################################################################################
def main(args):
    ### Stage 1
    cfg, eval_dataset, eval_db_vars, new_feat_path, new_feat_center, new_json_path = stage1infer_extractFeature(args)

    if args.config2 and not args.only_perfect:
        twotower_stage2eval(args, eval_dataset, eval_db_vars, new_feat_path, new_feat_center, new_json_path)

    if args.config2 and args.infer_perfect_stage1:
        print(f"Infer perfect stage 1 ...")
        # if cache exists, load it
        save_cache_name = os.path.basename(args.ckpt).split(".pth.tar")[0] + "_perfect.pkl"
        save_cache_path = os.path.join(args.perfect_stage1, save_cache_name)
        print(f"Try to load cache from {save_cache_path}")
        if os.path.isfile(save_cache_path):
            print(f"Loaded cache from {save_cache_path}")
            with open(save_cache_path, 'rb') as f:
                cache = pickle.load(f)
            perfect_result = cache['result']
            perfect_feat_center = cache['new_feat_center']
            perfect_json_path = cache['new_json_path']
        else:
            # re-extract features
            # do not support crop features
            assert args.re_extract == True
            ...

        print("Evaluating based on perfect stage 1 ...")
        twotower_stage2eval(
            args, eval_dataset, eval_db_vars, args.perfect_stage1, perfect_feat_center, perfect_json_path
        )


def twotower_stage2eval(args, eval_dataset, eval_db_vars, new_feat_path, new_feat_center, new_json_path):
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
        raise ValueError(f"Config file {args.config2} does not exist.")
    if os.path.isfile(args.config3):
        cfg2 = load_config(args.config3)
    else:
        raise ValueError(f"Config file {args.config3} does not exist.")
    cfg['raise_error'] = cfg2['raise_error'] = args.raise_error
    args.saveonly = False
    cfg['dataset']['feat_folder'] = new_feat_path # cache dir
    cfg2['dataset']['feat_folder'] = args.heatmap_dir
    cfg['dataset']['json_file'] = new_json_path
    cfg2['dataset']['json_file'] = new_json_path
    cfg['val_split'] = ['test']
    cfg2['val_split'] = ['test']
    if args.desired_actions is not None:
        cfg['dataset']['desired_actions'] = args.desired_actions
    pprint(cfg)
    pprint(cfg2)
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
                # debug
            actions = cfg['dataset']['desired_actions']
            print(f'Desired Action: {actions}')
            cfg2['dataset']['desired_actions'] = [action]
            result = run2tower(cfg, cfg2, args, action_label=action)

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
        results = run2tower(cfg, cfg2, args, action_label=cfg['dataset']['desired_actions'])
        results['seg-id'] = results['video-id']
        results['video-id'] = [i.split("#")[0] for i in results['video-id']]
        # shift t-start and t-end based on stage 1 segment results (+ t-center - 2)
    results = shift_result(new_feat_center, results, args.seg_duration)
        # evaluate
    mAP = det_eval.evaluate(results) # should be evaluated on the original video rather than the clipped video
    print(f"mAP: {mAP[0] * 100}")
    if args.eval_single_cls:
        activity_index = mAP[-1]
        ap = mAP[-2]
        for action, i in activity_index.items():
            mean_ap = ap[:, i].mean()
            print(f"Action: {action}, AP: {ap[:, i]}, Mean AP: {mean_ap}")
    if args.result_path:
        dump_result_path = os.path.join(args.result_path, 'final_result.pkl')
        os.makedirs(args.result_path, exist_ok=True)
        with open(dump_result_path, 'wb') as f:
            pickle.dump(results, f)

def get_args():
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR',default='configs/2stage/mamba_swallow_i3d_eval_stage1.yaml',
                        help='path to a config file, stage 1')
    parser.add_argument('--config2', metavar='DIR',
                        help='path to a config file, stage 2 tower 1')
    parser.add_argument('--config3', metavar='DIR',
                        help='path to a config filem stage 2 tower 2')
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
    parser.add_argument("--train_set", action='store_true', help="eval train set")
    parser.add_argument("--re-extract", action='store_true', help="whether to re-extract features at stage 2")
    parser.add_argument('--video_root', type=str, metavar='DIR', help='path to video root (specific for stage 2)')
    parser.add_argument('--cache_dir', type=str, metavar='DIR', default='./cache', help='path to cache dir for the extracted features (specific for stage 2)')
    parser.add_argument('--flow_dir', type=str, metavar='DIR', help='path to flow dir (specific when re-extract)')
    parser.add_argument("--flow_i3d", type=str, metavar='DIR', default='./pretrained/flow_imagenet.pt', help='path to flow i3d model')
    parser.add_argument("--rgb_i3d", type=str, metavar='DIR', default='./pretrained/pretrained_swallow_i3d.pth', help='path to rgb i3d model')
    parser.add_argument("--raise_error", action='store_true', help="raise error when video reading error")
    parser.add_argument("--test_first_stage", action='store_true', help="test first stage on AllTime")
    parser.add_argument("--heatmap", action='store_true', help="use heatmap as input")
    parser.add_argument("--heatmap_dir", type=str, metavar='DIR', default='tmp/heatmap', help='path to heatmap dir')
    parser.add_argument('--ckpt2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to checkpoints for stage 2')
    parser.add_argument('--tower_name', default='LogitsAvg', type=str,
                        help='name of the two-tower model (default: Convfusion)')
    parser.add_argument('--confidence', default=0.23, type=float,
                        help='confidence threshold for stage 1')
    parser.add_argument('--cropped_videos', action='store_true', help='the heatmap input (video_root) are cropped videos')
    parser.add_argument('--heatmap_branch', choices=['rgb', 'flow', 'none'], default='none', help='which branch to extract heatmap features')
    parser.add_argument('--heatmap_size', type=int, default=56, help='size of heatmap')
    parser.add_argument("--seg_duration", type=float, default=4.004, help='segment duration for stage 2 (default: 4.004)')
    parser.add_argument('--image_size', type=int, default=128, help='size of image')
    parser.add_argument('--heatmap_sigma', type=float, default=4, help='Heatmap sigma (default 4)')
    parser.add_argument("--result_path", type=str, default=None, help='path to save the result')
    parser.add_argument("--last_epoch", action='store_true', help="use the last epoch to evaluate (default: best epoch)")
    parser.add_argument("--infer_perfect_stage1", action='store_true', help="infer on best stage 1")
    parser.add_argument("--perfect_stage1", type=str, metavar='DIR', default='', help='path to extracted features')
    parser.add_argument("--only_perfect", action='store_true', help="only infer on perfect stage 1")
    parser.add_argument("--heatmap_type", type=str, default='fusion', choices=['fusion', 'keypoint', 'line'], help='heatmap type')
    parser.add_argument('--kalman', choices=['true', 'false','True','False'], default='true', help='use kalman to extract heatmaps')
    parser.add_argument('--normal_kalman', action='store_true', default=False)
    parser.add_argument('--selected_index', type=int, default=-1, help='selected index for heatmap feature')
    parser.add_argument('--calflops', action='store_true', help='calculate flops')
    parser.add_argument('--vw', type=float, default=None)
    parser.add_argument('--desired_actions', nargs='+', default=None)
    parser.add_argument('--eval_single_cls', action='store_true', help='evaluate single class action')
    args = parser.parse_args()
    return args

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    args = get_args()
    main(args)
