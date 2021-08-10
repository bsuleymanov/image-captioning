import copy
import numpy as np
import torch
from torch import nn

from transformer_layers import (
    PositionalEncoding, MultiHeadAttention
)


class CaptioningTransformer(nn.Module):
    def __init__(self, word_to_idx, input_dim, wordvec_dim,
                 n_heads=4, n_layers=2, max_len=50):
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_proj = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_len)
        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, n_heads=n_heads)
        self.transformer = TransformerDecoder(decoder_layer, n_layers=n_layers)
        self.apply(self._init_weights) # initialize each submodule
        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        batch_size, seq_len = captions.size()

        caption_emb = self.embedding(captions)
        caption_emb = self.positional_encoding(caption_emb)

        projected_features = self.visual_proj(features).unsqueeze(1)
        target_mask = torch.tril(
            torch.ones(seq_len, seq_len,
                       device=caption_emb.device,
                       dtype=caption_emb.dtype))
        features = self.transformer(target=caption_emb,
                                    memory=projected_features,
                                    target_mask=target_mask)
        scores = self.output(features)

        return scores

    def sample(self, features, max_len=30):
        with torch.no_grad():
            features = torch.tensor(features)
            batch_size = features.size(0)

            captions = self._null * np.ones((batch_size, max_len), dtype=np.int32)

            partial_caption = self._start * np.ones(batch_size, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_len):
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                word = torch.argmax(output_logits, dim=1)

                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, dense_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, n_heads, dropout_rate)
        self.enc_attn = MultiHeadAttention(input_dim, n_heads, dropout_rate)
        self.linear_output = nn.Sequential(
            nn.Linear(input_dim, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, input_dim))
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, target, memory, target_mask=None):
        target_add = self.self_attn(query=target, key=target,
                                 value=target, attn_mask=target_mask)
        target = target + self.dropout1(target_add)
        target = self.norm1(target)

        target_add = self.enc_attn(query=target, key=memory, value=memory)
        target = target + self.dropout2(target_add)
        target = self.norm2(target)

        target_add = self.linear_output(target)
        target = target + self.dropout3(target_add)
        target = self.norm3(target)

        return target

def clones(module, n_clones):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layers):
        super().__init__()
        self.layers = clones(decoder_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, target, memory, target_mask):
        output = target

        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask)

        return output