"""
Class to combine feed_forward_nn.py and multi_headed_self_attention.py
 to implement a Decoder-only Transformer block. 
"""

import torch 
import torch.nn as nn
import math
import models.multi_headed_self_attention as mhsa
import models.feed_forward_nn as ffnn

class Block(nn.Module):
    def __init__(self, 
                 d,
                 H,
                 T,
                 bias=False,
                 dropout=0.2,
    ):
        """
        Arguments:
            d: embedding dimension
            H: num attention heads 
            T: maximum length of token sequence 
            bias: whether to apply bias on linear layers 
            dropout: probability of dropout
        """
        self.ln1 = nn.LayerNorm(d)
        self.attn = mhsa.causal_self_attention(d, H, T, bias, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffnn == ffnn.FFNN(d, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        x = x + self.ffnn(self.ln2(x))

        return x


