import math
import configs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()

        # assert (d_k * n_heads == d_model)
        # assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)
        # self.jei = nn.Linear(d_model, d_k * n_heads)

        self.fc = nn.Linear(d_k * n_heads, d_model)

    def forward(self, q, k, v, j, mask=None):
        q = self.query(q) # input q(N x T x d_model)  output q(N x T x (h*d_k))
        k = self.key(k)
        v = self.value(v)
        # j = self.jei(j)

        N = q.shape[0]
        T = q.shape[1]

        # change the shape to (N, T, h, d_k) -> (N, h, T, d_k)
        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        # j = j.view(N, T, self.n_heads, self.d_k).transpose(1, 2) # just experiment =)

        # (q)(N , h, T, d_k) x (k)(N, h, d_k, T) --> (N, h, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        # attn_scores_j = q @ j.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
            # attn_scores_j = attn_scores_j.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_scores_j = F.softmax(attn_scores_j, dim=-1)

        # compute attention-weighted values
        # (N, h, T, T) x (N, h, T, d_k) -> (N, h, T, d_k)
        # sub_a = attn_weights @ attn_scores_j
        A = attn_weights @ v

        # reshape back (N, T, h, d_k)
        A = A.transpose(1, 2)

        # (N, T, h*d_k)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads)

        return self.fc(A)


class TransformerBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_prob))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.mha(x, x, x, mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, max_len, d_k, d_model,
        n_heads, n_layers, n_classes, dropout_prob
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformers_blocks = [
            TransformerBlock(d_k, d_model, n_heads, dropout_prob)
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformers_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x, mask)

        x = x[:, 0, :]
        x = self.ln(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":

    model = Encoder(configs.VOCAB_SIZE, configs.MAX_LENGTH, configs.QK_SIZE,
                    configs.EMBED_SIZE, configs.N_HEADS, configs.N_LAYERS,
                    configs.NUM_CLASSES, configs.DROPOUT)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    x = np.random.randint(0, 20000, size=(8, 512))
    x_t = torch.tensor(x).to(device)
    mask = np.ones((8, 512))
    mask[:, 256:] = 0
    mask_t = torch.tensor(mask).to(device)
    y = model(x_t, mask_t)
    print(y)
