import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0

        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        denom = 10000 ** (torch.arange(0, embed_dim, 2) / embed_dim)
        pe[:, :, ::2] = torch.sin(position / denom)
        pe[:, :, 1::2] = torch.cos(position / denom)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        batch_size, seq_len, dim = x.size()
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((batch_size, seq_len, dim))

        output = x + self.pe[:, :seq_len, :]
        output = self.dropout(output)

        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.dim_per_head = embed_dim // num_heads
        self.scale = torch.sqrt(torch.tensor([self.dim_per_head],
                                             dtype=torch.float32))

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        batch_size, enc_len, dim = query.size()
        _, dec_len, _ = value.size()
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((batch_size, dec_len, dim))
        queries = self.query(query).view(batch_size, enc_len,
                                         self.num_heads, self.dim_per_head)\
                                   .permute(0, 2, 1, 3)
        keys = self.key(key).view(batch_size, dec_len,
                                  self.num_heads, self.dim_per_head)\
                            .permute(0, 2, 1, 3)
        values = self.value(value).view(batch_size, dec_len,
                                        self.num_heads, self.dim_per_head) \
                                  .permute(0, 2, 1, 3)
        scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.scale
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -float("inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, values)\
                               .permute(0, 2, 1, 3).contiguous()\
                               .view(batch_size, -1, dim)
        output = self.proj(output)


        return output