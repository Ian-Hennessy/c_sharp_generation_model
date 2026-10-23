"""
Module to implement the residual stream, 'carrying over' data 
from previous layers into the next layer to assist in widening context 
and permitting Decoder to capture more rich textual patterns. 

Helpful for Coding models because often programming requires 
knowledge of previously declared functions and variables, as well as a wide 
window of context for organization and structure of program. 

"""
import torch
import torch.nn as nn 
import math 


class ResidualStreamBlock(nn.Module):
    def __init__(self, size, dropout):
        """
        Initializes the ResidualStreamBlock module with a specific size for normalization
        and a specified dropout rate.
        Args:
            size (int): The number of features in the input tensors expected by the layer normalization.
            dropout (float): The dropout probability to be used in the dropout layer.
        """
        super().__init__()

        self.norm = nn.LayerNorm(size) # normalizes inputs to size

        self.dropout = nn.Dropout(dropout) # dropout block for regularization

        self.size = size 

    def forward(self, x, sublayer):
        """
        Run one forward pass of the ResidualStreamBlock. 
        1. Apply residual connection
        2. Dropout after the sublayer is applied to normalized tensor, 
        *before* any skip connections
        """

        residual = x # store raw x for residual passing to next sublayer


        normed = self.norm(x)

        # call sublayer on normed:

        sub_out = sublayer(normed)

        dropped = self.dropout(sub_out)

        out = residual + dropped # apply the residual connection to the sublayer's output after dropout applied 

        return out