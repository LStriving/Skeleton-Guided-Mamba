# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(args.ckpt,
                                     'epoch_{:03d}.pth.tar'.format(args.epoch))
        else:
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)
    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)
    """2. create dataset / dataloader"""
    if args.train_set:
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
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.result_path:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            only_focus_on=cfg['dataset']['desired_actions']
        )
    else:
        output_file = os.path.join(
            args.result_path, os.path.split(ckpt_file)[0], f'eval_results.pkl')
    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(val_loader,
                          model,
                          -1,
                          evaluator=det_eval,
                          output_file=output_file,
                          ext_score_file=cfg['test_cfg']['ext_score_file'],
                          tb_writer=None,
                          print_freq=args.print_freq,
                          visualize=args.visualize,
                          gt_file=val_dataset.data_list)
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument(
        '--config',
        type=str,
        metavar='DIR',
        default='./configs/exp/surgery_i3d_rgb_shuffle_w37.yaml',
        help='path to a config file')
    parser.add_argument('--ckpt',
                        type=str,
                        metavar='DIR',
                        default='ckpt/surgery_i3d_rgb_shuffle_w37_m9216',
                        help='path to a checkpoint')
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t',
                        '--topk',
                        default=-1,
                        type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument(
        '--result_path',
        type=str, default=None, help='output result path')
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
    args = parser.parse_args()
    main(args)
