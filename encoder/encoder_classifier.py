import math
import configs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import numpy as np
import lightning.pytorch as pl
from clean_prep import clean_prepare, tokenizer
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

logger = TensorBoardLogger("tb_logs", name="encoder_classifier")


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


class Encoder(pl.LightningModule):
    def __init__(
        self, vocab_size, max_len, d_k, d_model,
        n_heads, n_layers, n_classes, dropout_prob
    ):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=configs.NUM_CLASSES)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=configs.NUM_CLASSES)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformers_blocks = [
            TransformerBlock(d_k, d_model, n_heads, dropout_prob)
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformers_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = x[:, 0, :]
        x = self.ln(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self.forward(x['input_ids'], x['attention_mask'])
        loss = self.loss(outputs, x['labels'])
        acc = self.train_acc(outputs, x['labels'])

        self.log_dict({'train_loss': loss, 'train_accuracy': acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch
        outputs = self.forward(y['input_ids'], y['attention_mask'])
        loss = self.loss(outputs, y['labels'])
        acc = self.valid_acc(outputs, y['labels'])
        self.log_dict({'val_loss': loss, 'val_accuracy': acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    pass
    # model = Encoder(
    #     vocab_size=tokenizer.vocab_size,
    #     max_len=configs.MAX_LENGTH,
    #     d_k=configs.QK_SIZE,
    #     d_model=configs.EMBED_SIZE,
    #     n_heads=configs.N_HEADS,
    #     n_layers=configs.N_LAYERS,
    #     n_classes=configs.NUM_CLASSES,
    #     dropout_prob=configs.DROPOUT)
    #
    # trainer = Trainer(
    #     callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    #     max_epochs=10,
    #     min_epochs=3,
    #     logger=model.logger)
    #
    # train_loader, valid_loader = clean_prepare()
    # trainer.fit(model, train_loader, valid_loader)
