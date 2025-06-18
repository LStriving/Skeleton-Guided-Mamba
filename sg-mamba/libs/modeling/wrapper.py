import math
import torch.nn as nn

class Wrapper(nn.Module):
    """
    A wrapper class for nn.Module for reducing the memory usage (with more forward passes)
    """
    def __init__(self, module, chunk_size):
        super().__init__()
        self.module = module
        self.chunk_size = chunk_size
    
    def forward(self, x):
        # slice the input into smaller chunks
        chunk_num = math.ceil(x.size(0) / self.chunk_size)
        x_chunks = x.chunk(chunk_num, dim=0)
        outputs = []
        for  x_chunk in x_chunks:
            x = self.module(x_chunk)
            outputs.append(x)
        # concatenate the outputs
        x = torch.cat(outputs, dim=0)
        return x