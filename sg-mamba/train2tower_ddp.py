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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

def setup(rank, world_size):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理DDP进程组"""
    dist.destroy_process_group()

def run(cfg, cfg2, args, action_label=None, rank=0, world_size=1):
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
    if rank == 0 and not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    
    # 确保所有进程等待主进程创建目录
    if world_size > 1:
        dist.barrier()
    
    # tensorboard writer (只在主进程上创建)
    tb_writer = None
    if rank == 0:
        tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'] + rank, include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= world_size
    cfg['loader']['num_workers'] = max(1, cfg['loader']['num_workers'] // world_size)
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
    
    # 创建DDP采样器
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            mulmodal_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        train_sampler = None
    
    # 修改数据加载器以使用DDP采样器
    train_loader = make_data_loader(
        mulmodal_dataset, True, rng_generator, sampler=train_sampler, **cfg['loader'])
    
    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])

    """3.5 Initialize (partially) from pre-trained model"""
    if args.backbone_1:
        if os.path.isfile(args.backbone_1):
            checkpoint = torch.load(args.backbone_1,
                map_location = lambda storage, loc: storage.cuda(rank))
            new_kv = {}
            for k,v in checkpoint['state_dict_ema'].items():
                new_kv[k.replace("module.","")]=v
            model.load_state_dict(new_kv)
            if rank == 0:
                print("=> loaded checkpoint '{:s}' for tower 1".format(args.backbone_1))
            del checkpoint
        else:
            if rank == 0:
                print("=> no checkpoint found at '{}'".format(args.backbone_1))
            return
    if args.backbone_2:
        if os.path.isfile(args.backbone_2):
            checkpoint = torch.load(args.backbone_2,
                map_location = lambda storage, loc: storage.cuda(rank))
            new_kv = {}
            for k,v in checkpoint['state_dict_ema'].items():
                new_kv[k.replace("module.","")]=v
            if args.filter_backbone2:
                new_kv = {k:v for k,v in new_kv.items() if args.filter_backbone2 in k}
                model2.load_state_dict(new_kv, strict=False)
            else:
                model2.load_state_dict(new_kv)
            if rank == 0:
                print("=> loaded checkpoint '{:s}' for tower 2".format(args.backbone_2))
            del checkpoint
        else:
            if rank == 0:
                print("=> no checkpoint found at '{}'".format(args.backbone_2))


    # two-tower model
    model = make_two_tower(args.tower_name, model, model2, cfg, cfg2, **cfg['two_tower'])

    # 将模型移动到对应设备
    torch.cuda.set_device(rank)
    model.cuda(rank)
    
    # 使用DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'], args.filter_backbone2, args.lower_ckpt_lr_rate)
    # schedule
    num_iters_per_epoch = len(train_loader) // cfg['loader']['accum_steps'] 
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    if rank == 0:
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
                map_location = lambda storage, loc: storage.cuda(rank))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                    ckpt_file, checkpoint['epoch']
                ))
            del checkpoint

    # save the current config (只在主进程上保存)
    if rank == 0:
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint('config 1:', fid)
            pprint(cfg, stream=fid)
            pprint('config 2:', fid)
            pprint(cfg2, stream=fid)
            fid.flush()

    """5. training / validation loop"""
    if rank == 0:
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
    
    # 验证集不需要分布式采样器
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
            if rank == 0:
                print(f"Warning: {label} not found in eval_label_dict")
            remap = True
            break

    for epoch in range(args.start_epoch, max_epochs):
        # 设置采样器的epoch
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer if rank == 0 else None,
            print_freq=args.print_freq,
            accum_step_num=cfg['loader']['accum_steps'],
            devices=[rank],
            rank=rank,
            world_size=world_size
        )
        
        start_eval = 5 if max_epochs > 30 else 0

        # 只在主进程上进行评估
        if rank == 0 and (epoch>=start_eval or not cfg['opt']['warmup']):
            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            model_eval2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])
            print(f'{args.tower_name} model loaded for evaluation')
            model_eval = make_two_tower(args.tower_name, model_eval, model_eval2, cfg, cfg2, **cfg['two_tower'])
            # 移动到GPU
            model_eval = nn.DataParallel(model_eval, device_ids=['cuda:0'])
            # 加载EMA模型权重
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

            # 保存模型（只在主进程上）
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

    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
        
    # wrap up
    if rank == 0 and tb_writer is not None:
        tb_writer.close()
        print("All done!")
    
    # 清理进程组
    if world_size > 1:
        cleanup()
    
    return

def run_ddp(rank, world_size, cfg, cfg2, args, action_label=None):
    """DDP进程入口函数"""
    # 初始化DDP
    if world_size > 1:
        setup(rank, world_size)
    
    # 设置设备
    cfg['devices'] = [rank]
    cfg2['devices'] = [rank]
    
    # 运行训练
    run(cfg, cfg2, args, action_label, rank, world_size)

################################################################################
def main(args):
    """main function that handles training / inference"""
    cfg = get_cfg(args.config)
    cfg2 = get_cfg(args.config2)
    
    cfg['devices'] = ['cpu']
    cfg2['devices'] = ['cpu']
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

    # 获取世界大小（GPU数量）
    world_size = len(cfg['devices'])
    
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

        # 使用多进程启动多个动作的训练
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
                
            # 为每个动作创建一个新的配置副本
            action_cfg = cfg.copy()
            action_cfg2 = cfg2.copy()
            action_cfg['dataset']['desired_actions'] = [action]
            action_cfg2['dataset']['desired_actions'] = [action]
            
            # 设置输出目录
            output_prefix = f'{action}_'
            action_args = argparse.Namespace(**vars(args))
            action_args.output = f'{output_prefix}{output}'
            
            # 启动DDP训练
            if world_size > 1:
                mp.spawn(
                    run_ddp,
                    args=(world_size, action_cfg, action_cfg2, action_args, action),
                    nprocs=world_size,
                    join=True
                )
            else:
                # 单GPU情况直接运行
                run(action_cfg, action_cfg2, action_args, action, 0, 1)
    else:
        # 多类别情况
        if world_size > 1:
            mp.spawn(
                run_ddp,
                args=(world_size, cfg, cfg2, args, action_label),
                nprocs=world_size,
                join=True
            )
        else:
            # 单GPU情况直接运行
            run(cfg, cfg2, args, action_label, 0, 1)

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
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu instead of gpu')
    parser.add_argument('--enable_branch_eval', action='store_true',
                        help='enable evaluation for each branch')
    parser.add_argument('--filter_backbone2', default=None, type=str,
                        help='filter backbone 2')
    parser.add_argument("--lower_ckpt_lr_rate", type=float, default=1.0, help="lr ratio for the ckpt part (default: 1.0)")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    args = parser.parse_args()
    main(args)
