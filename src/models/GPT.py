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

class GPT(nn.Module):
    def __init__(self,
                 d,
                 H,
                 T,
                 V,
                 layers,
                 bias=False,
                 dropout=0.2
                 ):
        """
        Arguments:
            d: size of embedding dimension
            H: number of heads 
            T: maximum token sequence length 
            V: size of the token vocabulary 
            layers: number of decoder-only blocks 
            bias: whether to apply bias to linear layers 
            dropout: probability of dropout
        """
        super().__init__()
        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(V, d),
            wpe = nn.Embedding(T, d),
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([block.block(d, H, T, bias, dropout)] for _ in range(layers)),
            ln_f = nn.LayerNorm(d),
            head = nn.Linear(d, V, bias=bias)
            )

    def forward(self, idx, targets=None):

        
