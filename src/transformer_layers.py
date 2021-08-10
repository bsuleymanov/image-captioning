import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
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
        batch_size, seq_len, dim = x.size()

        output = x + self.pe[:, :seq_len, :]
        output = self.dropout(output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.dim_per_head = embed_dim // n_heads
        self.register_buffer(
            'scale', torch.sqrt(torch.tensor([self.dim_per_head],
                                             dtype=torch.float32))
        )

    def forward(self, query, key, value, attn_mask=None):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query, key, value = map(lambda t: rearrange(t, "b n (h d) -> b h n d",
                                                    h=self.n_heads),
                                [query, key, value])
        logits = einsum("b h i d, b h j d -> b h i j", query, key) / self.scale
        if attn_mask is not None:
            logits.masked_fill_(attn_mask == 0, -float("inf"))
        attn_weights = self.dropout(torch.softmax(logits, dim=-1))
        output = einsum("b h i j, b h j d -> b h i d", attn_weights, value)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.proj(output)

        return output