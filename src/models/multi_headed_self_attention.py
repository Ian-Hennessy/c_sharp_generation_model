"""
The multiheaded self attention model presents a self attention block for the decoder-only 
transformer architecture, using masked self-attention. 
"""

import math
import torch
import torch.nn as nn

import torch.nn.functional as F


class causal_self_attention(nn.Module):
    def __init__(self, 
                 d, 
                 H,
                 T,
                 bias=False,
                 dropout=0.2):
        """ Arguments:
        d = embed dimensions
        H = # attention heads 
        T = maximum sequence length (Tokens)
        bias = whether or not to include bias in linear layers 
        dropout = prob of dropout 

        """
        super().__init__()
        # 
        # ensure the number of heads can fit the dimensions of the embedding 
        assert d % H == 0

        # key, val, query embeddings batches for all heads
        # output = 3x the original embedding dim

        self.c_attn = nn.Linear(d, 3*d, bias=bias)

        # project the concatenated head outputs 
        self.c_projection = nn.Linear(d, d, bias=bias)

        # implement dropout 

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to apply only the preceding tokens to attention score 
        # sets all values above the diagnonal to -inf

        self.register_buffer("mask", torch.tril(torch.ones(T,T),).view(1,1,T,T))

    def forward(self, x):
        """
        Arguments: 
            x: embedded data input
        """
        B, T, _ = x.size() # batch size, sequence length, embedding dimension

        # compute query, key, val vectors for all heads in a batch 
        # split output into separate query, value key vectors 
        q, k, v = self.c_attn(x).split(self.d, dim = 2) # [B, T, D]

        # reshape vectors into sequences of smaller token vectors for heads 

        k = k.view(B, T, self.d // self.H).transpose(1, 2) # B, T, d // h
        q = q.view(B, T, self.d // self.H).transpose(1, 2) # B, T, d // h
        v = v.view(B, T, self.d // self.H).transpose(1, 2)

        # compoute attention matris 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(2)) # [B, H, T, T]

        att = att.masked_fill(self.mask[:, :, T, T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)

        att = self.attn_dropout(att)

        # compute output vectors 
        y = att @ v # [B, H, T, d // H]

        y = y.transpose(1, 2).contiguous.view(B, T, self.d)

        y = self.resid_dropout(self.c_proj(y))
        return y

