# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from eval2stage import get_best_pth_from_dir
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch, make_two_tower
from libs.utils import (train_one_epoch, valid_one_epoch, infer_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from train2stage import get_label_dict_from_file
from libs.datasets.swallow import MultiModalDataset

def get_cfg(config_file):
    if os.path.isfile(config_file):
        cfg = load_config(config_file)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    return cfg

def run(cfg, cfg2, args, action_label=None):
    """1. get configuration from a yaml file"""
    args.start_epoch = 0
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'], exist_ok=True)
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    cfg2_filename = os.path.basename(args.config2).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + cfg2_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + cfg2_filename + '_' + \
                str(args.output)+'_'+str(cfg['loader']['batch_size'])+'_'+str(cfg['opt']['learning_rate']))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    cfg['loader']['accum_steps'] = 1 if cfg['loader']['accum_steps'] <= 0 else cfg['loader']['accum_steps']
    
    cfg2['video_stem']['num_frames'] = cfg2['dataset']['num_frames']
    
    """2. create dataset / dataloader"""
    # train dataset for tower 1
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'], 
    )
    # dataset for tower 2
    train_dataset2 = make_dataset(
        cfg2['dataset_name'], True, cfg2['train_split'], **cfg2['dataset'], 
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars2 = train_dataset2.get_attributes()
    cfg2['model']['train_cfg']['head_empty_cls'] = train_db_vars2['empty_label_ids']
    # multi-modal dataloader
    mulmodal_dataset = MultiModalDataset(train_dataset, train_dataset2)
    
    # data loaders
    train_loader = make_data_loader(
        mulmodal_dataset, True, rng_generator, **cfg['loader'])
    
    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])

    """3.5 Initialize (partially) from pre-trained model"""
    if args.backbone_1:
        if os.path.isfile(args.backbone_1):
            checkpoint = torch.load(args.backbone_1,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            new_kv = {}
            for k,v in checkpoint['state_dict_ema'].items():
                new_kv[k.replace("module.","")]=v
            model.load_state_dict(new_kv)
            print("=> loaded checkpoint '{:s}' for tower 1".format(args.backbone_1))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.backbone_1))
            return
    if args.backbone_2:
        if os.path.isfile(args.backbone_2):
            checkpoint = torch.load(args.backbone_2,
                map_location = lambda storage, loc: storage.cuda(
                    cfg2['devices'][0]))
            new_kv = {}
            for k,v in checkpoint['state_dict_ema'].items():
                new_kv[k.replace("module.","")]=v
            if args.filter_backbone2:
                new_kv = {k:v for k,v in new_kv.items() if args.filter_backbone2 in k}
                model2.load_state_dict(new_kv, strict=False)
            else:
                model2.load_state_dict(new_kv)
            print("=> loaded checkpoint '{:s}' for tower 2".format(args.backbone_2))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.backbone_2))


    # two-tower model
    model = make_two_tower(args.tower_name, model, model2, cfg, cfg2, **cfg['two_tower'])

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'], args.filter_backbone2, args.lower_ckpt_lr_rate)
    # schedule
    num_iters_per_epoch = len(train_loader) // cfg['loader']['accum_steps'] 
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        files = [f for f in os.listdir(ckpt_folder) if f.endswith('.pth.tar')]
        ckpt_file_list = sorted(files, key=lambda x: int(x.split('.pth')[0].split('_')[1]))
        ckpt_file = None
        if len(ckpt_file_list) > 0:
            ckpt_file = os.path.join(ckpt_folder, ckpt_file_list[-1]) # latest ckpt

        if os.path.isfile(args.resume):
            ckpt_file = args.resume

        if ckpt_file is not None:
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(ckpt_file,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                ckpt_file, checkpoint['epoch']
            ))
            del checkpoint

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint('config 1:', fid)
        pprint(cfg, stream=fid)
        pprint('config 2:', fid)
        pprint(cfg2, stream=fid)
        fid.flush()

    """5. training / validation loop"""
    print("\nStart training model {:s} ...".format(args.tower_name))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    """6. create val dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_dataset2 = make_dataset(
        cfg2['dataset_name'], False, cfg2['val_split'], **cfg2['dataset']
    )
    valMultidataset = MultiModalDataset(val_dataset, val_dataset2)
    cfg['loader']['batch_size'] = 1
    val_loader = make_data_loader(
        valMultidataset, False, rng_generator, **cfg['loader']
    )
    
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds'],
                only_focus_on = action_label
            )
    train_label_dict = val_dataset.label_dict
    eval_label_dict = get_label_dict_from_file(val_dataset.json_file, action_label)
    remap = False
    for label, train_label_id in train_label_dict.items():
        if label in eval_label_dict:
            if train_label_id != eval_label_dict[label]:
                remap = True
                break
        else:
            print(f"Warning: {label} not found in eval_label_dict")
            remap = True
            break

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            accum_step_num=cfg['loader']['accum_steps'],
            devices=cfg['devices']
        )
        
        start_eval = 5 if max_epochs > 30 else 0

        if epoch>=start_eval or not cfg['opt']['warmup']:#(max_epochs//4):
            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            model_eval2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])
            print(f'{args.tower_name} model loaded for evaluation')
            model_eval = make_two_tower(args.tower_name, model_eval, model_eval2, cfg, cfg2, **cfg['two_tower'])
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg['devices'])
            model_eval.load_state_dict(model_ema.module.state_dict())

            # set up evaluator
            output_file = None
            
            """5. Test the model"""
            print("\nStart testing model {:s} ...".format(args.tower_name))
            start = time.time()
            result = infer_one_epoch(
                val_loader,
                model_eval,
                -1,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writer,
                print_freq=999999 #args.print_freq
            )
            # remap action labels
            if remap:
                for label, train_label_id in train_label_dict.items():
                    if label in eval_label_dict:
                        result['label'][result['label'] == train_label_id] = eval_label_dict[label] + 1000
                    else:
                        print(f"Warning: {label} not found in eval_label_dict")
                result['label'] -= 1000
            _, mAP = det_eval.evaluate(result)
            if tb_writer is not None:
                tb_writer.add_scalar('validation/mAP', mAP, epoch)
            end = time.time()
            # print("All done! Total time: {:0.2f} sec".format(end - start))
            print(epoch,mAP)

            if args.enable_branch_eval:
                test_vws = [0.0, 1.0]
                for vw in test_vws:
                    model_eval.module.vw = vw
                    print(f"Start testing model {args.tower_name} with vw={vw} ...")
                    result = infer_one_epoch(
                        val_loader,
                        model_eval,
                        -1,
                        evaluator=det_eval,
                        output_file=output_file,
                        ext_score_file=cfg['test_cfg']['ext_score_file'],
                        tb_writer=tb_writer,
                        print_freq=999999 #args.print_freq
                    )
                    # remap action labels
                    if remap:
                        for label, train_label_id in train_label_dict.items():
                            if label in eval_label_dict:
                                result['label'][result['label'] == train_label_id] = eval_label_dict[label] + 1000
                            else:
                                print(f"Warning: {label} not found in eval_label_dict")
                        result['label'] -= 1000
                    _, mAP = det_eval.evaluate(result)
                    if tb_writer is not None:
                        tb_writer.add_scalar(f'validation/vw{vw}_mAP', mAP, epoch)

            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}_{:.5f}.pth.tar'.format(epoch,mAP)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
def main(args):
    from torch.multiprocessing import Process, set_start_method
    """main function that handles training / inference"""
    cfg = get_cfg(args.config)
    cfg2 = get_cfg(args.config2)
    
    if args.init_rand_seed is not None:
        cfg['init_rand_seed'] = args.init_rand_seed
        cfg2['init_rand_seed'] = args.init_rand_seed
    # get stage
    stage = cfg['dataset']['stage_at']
    assert stage in [1, 2], "Stage must be 1 or 2!"
    # get desired action label
    action_label = cfg['dataset']['desired_actions']
    if action_label is None and cfg2['dataset']['desired_actions'] is None:
        print(f'Using all actions')
    else:
        assert set(action_label) == set(cfg2['dataset']['desired_actions']),\
          "Action labels must be the same for two configs!"
    assert cfg['loader']['batch_size'] == cfg2['loader']['batch_size'],\
            "Batch size must be the same for two configs!"
    
    if stage == 1 and cfg['dataset']['two_stage']:
        assert len(action_label) == 1, "Stage 1 only supports one action label!"

    if cfg['dataset']['num_classes'] == 1:
        # looping over all actions
        output = args.output

        # load best backbone model
        backbone_1, backbone_2 = {}, {}
        if args.backbone_1:
            assert os.path.isdir(args.backbone_1), "Backbone 1 must be a directory!"
            ckpt_root_1 = os.listdir(args.backbone_1)

        if args.backbone_2:
            assert os.path.isdir(args.backbone_2), "Backbone 2 must be a directory!"
            ckpt_root_2 = os.listdir(args.backbone_2)

        ori_backbone_1 = args.backbone_1
        ori_backbone_2 = args.backbone_2

        set_start_method('spawn', force=True)
        processes = []
        for rank, action in enumerate(action_label):
            # modify the backbone path
            if ori_backbone_1:
                ckpt_dir_1 = [ckpt for ckpt in ckpt_root_1 if action in ckpt]
                if len(ckpt_dir_1) == 0:
                    print(f"Error: {action} ckpt not found for backbone 1!")
                    raise FileNotFoundError
                elif len(ckpt_dir_1) > 1:
                    print(f"Warning: multiple {action} ckpt found in backbone 1! Using the first one.")
                ckpt_dir_1 = ckpt_dir_1[0]
                best_ckpt = get_best_pth_from_dir(os.path.join(ori_backbone_1, ckpt_dir_1))
                args.backbone_1 = best_ckpt
            if ori_backbone_2:
                args.backbone_2 = os.path.join(ori_backbone_2, action)
                ckpt_dir_2 = [ckpt for ckpt in ckpt_root_2 if action in ckpt]
                if len(ckpt_dir_2) == 0:
                    print(f"Error: {action} ckpt not found for backbone 2!")
                    raise FileNotFoundError
                elif len(ckpt_dir_2) > 1:
                    print(f"Warning: multiple {action} ckpt found in backbone 2! Using the first one.")
                ckpt_dir_2 = ckpt_dir_2[0]
                best_ckpt = get_best_pth_from_dir(os.path.join(ori_backbone_2, ckpt_dir_2))
                args.backbone_2 = best_ckpt
            p = Process(target=train_action, args=(cfg, cfg2, args, output, action, rank))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        run(cfg, cfg2, args, action_label)

def train_action(cfg, cfg2, args, output, action, rank):
    output_prefix = f'{action}_'
    args.output = f'{output_prefix}{output}'
    cfg['dataset']['desired_actions'] = [action]
    cfg2['dataset']['desired_actions'] = [action]
    cfg['devices'] = [f'cuda:{rank}']
    cfg2['devices'] = [f'cuda:{rank}']
    run(cfg, cfg2, args, action)

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file, will use the training config')
    parser.add_argument('config2', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--backbone_1', default=None, type=str, metavar='PATH',
                        help='path to a checkpoint for tower 1(default: none)')
    parser.add_argument('--backbone_2', default=None, type=str, metavar='PATH',
                        help='path to a checkpoint for tower 2(default: none)')
    parser.add_argument('--tower_name', default='DINOAttnEarlyFusion', type=str,
                        help='name of the two-tower model (default: DINOAttnEarlyFusion)')
    parser.add_argument('--enable_branch_eval', action='store_true',
                        help='enable evaluation for each branch')
    parser.add_argument('--filter_backbone2', default=None, type=str,
                        help='filter backbone 2')
    parser.add_argument('--init_rand_seed', default=None, help='seed for reproduction', type=int)
    parser.add_argument("--lower_ckpt_lr_rate", type=float, default=1.0, help="lr ratio for the ckpt part (default: 1.0)")
    args = parser.parse_args()
    main(args)
