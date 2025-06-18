import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers, sampler=None, accum_steps=1):
    """
        A simple dataloder builder
    """
    if accum_steps > 0 and is_training:
        assert batch_size % accum_steps == 0, "Only support accum_steps that divides batch_size"
        batch_size = batch_size // accum_steps

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        sampler=sampler,
        shuffle=is_training and not sampler,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
