import os
import torch

def load_backbone(model, path, device):
    if os.path.isfile(path):
        print("=> loaded checkpoint '{:s}' for tower 1".format(path))
        # alter key value mapping:
        checkpoint = torch.load(path,
            map_location = lambda storage, loc: storage.cuda(
                device))
        # video mae key value map
        new_kv = {}
        for k,v in checkpoint['state_dict_ema'].items():
            new_kv[k.replace("module.","")]=v
        model.load_state_dict(new_kv)
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(path))
        return
    return model