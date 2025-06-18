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
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, infer_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

def get_cfg(args):
    """get configuration from a yaml file"""
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    return cfg

def run(cfg, args, action_label=None):
    args.start_epoch = 0
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output)+'_'+str(cfg['loader']['batch_size'])+'_'+str(cfg['opt']['learning_rate']))
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

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'], 
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader) // cfg['loader']['accum_steps']
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        # try to load from the last checkpoint
        ## get checkpoint file
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
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )


    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
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
            accum_step_num=cfg['loader']['accum_steps']
        )
        
        start_eval = 5 if max_epochs > 30 else 0

        if epoch>=start_eval or not cfg['opt']['warmup']:#(max_epochs//4):


        # if epoch>1:#(max_epochs//3):

            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg['devices'])


            model_eval.load_state_dict(model_ema.module.state_dict())


            # set up evaluator
            output_file = None
            # if not args.saveonly:
            val_db_vars = val_dataset.get_attributes()
            
            # else:
            #     output_file = os.path.join('eval_results.pkl')

            """5. Test the model"""
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
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
    cfg = get_cfg(args)
    
    # get stage
    stage = cfg['dataset']['stage_at']
    assert stage in [1, 2], "Stage must be 1 or 2!"
    # get desired action label
    action_label = cfg['dataset']['desired_actions']
    
    if stage == 1:
        assert len(action_label) == 1, "Stage 1 only supports one action label!"

    output = args.output
    if cfg['dataset']['num_classes'] == 1 and stage == 2:
        # looping over all actions
        set_start_method('spawn', force=True)
        processes = []
        for rank, action in enumerate(action_label):
            p = Process(target=train_action, args=(cfg, args, output, action, rank))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    elif stage == 2:
        run(cfg, args, action_label)
    else: # stage 1
        action = action_label[0]
        output_prefix = f'{action}_'
        args.output = f'{output_prefix}{output}'
        cfg['dataset']['desired_actions'] = [action]
        run(cfg, args, action_label)

def train_action(cfg, args, output, action, rank):
    num_gpus = torch.cuda.device_count()
    print(f"Training action: {action} on GPU {rank}")
    output_prefix = f'{action}_'
    args.output = f'{output_prefix}{output}'
    cfg['dataset']['desired_actions'] = [action]
    cfg['devices'] = [f'cuda:{rank}'] if rank < num_gpus else [f'cuda:{rank % num_gpus}']
    run(cfg, args, action)

def get_label_dict_from_file(json_file, action_label):
    import json
    with open(json_file, 'r') as fid:
        data = json.load(fid)
    if 'database' in data:
        data = data['database']
    label_dict = {}
    
    for _, v in data.items():
        for act in v['annotations']:
            if action_label is None:
                label_dict[act['label']] = act['label_id']
            else:
                if act['label'] in action_label:
                    label_dict[act['label']] = act['label_id']

    return label_dict

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
