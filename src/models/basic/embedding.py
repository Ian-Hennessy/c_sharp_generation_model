"""
Module to convert token indices to embeddings, scaled by square root of model dimension
"""

import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Embeddings, self).__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(model_dim, vocab_size)

    def forward(self, x):
        #1. Embed x
        x = self.embed(x)

        # scale by square root of model_dim
        x = x * math.sqrt(self.d_model)

        out = x # shape = [batch_size, sequence_length, model_dim]

        return x

    def set_embedding_weights(self):
        """
        Set the weights of the embedding layer.
        """
        for idx in range(self.vocab_size):
            self.embed.weight.data[idx] = torch.linspace(start=0.0, end=1.0, steps=self.d_model)
