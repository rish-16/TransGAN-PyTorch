import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper uses Vaswani (2017) Attention with minimal changes.

Multi-head self-attention with a feed-forward MLP
with GELU non-linearity. Layer normalisation is used
before each segment and employs residual skip connections.
"""
class Attention(nn.Module):
    def __init__(self, D, heads=8):
        super().__init__()
        self.D = D
        self.heads = heads

        assert (D % heads == 0), "Embedding size should be divisble by number of heads"
        self.head_dim = D // heads

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.FC_H = nn.Linear(D, D)

    def forward(self, Q, K, V, mask):
        batch_size = Q.shape[0]
        q_len, k_len, v_len = Q.shape[1], K.shape[1], V.shape[1]

        Q = Q.reshape(batch_size, q_len, self.heads, self.head_dim)
        K = K.reshape(batch_size, k_len, self.heads, self.head_dim)
        V = V.reshape(batch_size, v_len, self.heads, self.head_dim)

        # performing batch-wise matrix multiplication
        raw_scores = torch.einsum("bqhd,bkhd->bhqk", [Q, K])

        # shut off triangular matrix with very small value
        scores = raw_scores.masked_fill(mask == 0, -np.inf) if mask else raw_scores

        attn = torch.softmax(scores / np.sqrt(D), dim=3)
        attn_output = torch.einsum("bhql,blhd->bqhd", [attn, V])
        attn_output = attn_output.reshape(batch_size, q_len, D)

        output = self.FC_H(attn_output)

        return output

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass      
        
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        pass        

class TransGAN(nn.Module):
    def __init__(self):
        super().__init__()
        pass