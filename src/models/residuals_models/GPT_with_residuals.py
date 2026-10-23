"""
Implement Decoder-Only Generative Pretrained Transformer model 
using feed_forward_nn.py, block.py, multi_headed_self_attention.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import block
import feed_forward_nn as ffnn
import multi_headed_self_attention as mhsa
import residual_stream_block as rsb
import copy

def clones(module: nn.Module, count: int) -> nn.ModuleList:
    """
    Clones an nn.Module block [count] number of times and returns as a ModuleList.
    Helpful for deep networks with many identical layers 
    Args:
        module: The module to be cloned
        count: the number of the cloned modules to create

    Returns:
        nn.ModuleList modules: an nn.ModuleList object containing {count} modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])