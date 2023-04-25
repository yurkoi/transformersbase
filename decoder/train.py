from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from decoder import Decoder
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
dataset = load_dataset("math_dataset", "algebra__linear_2d")


def tokenize_fn(batch):
    return tokenizer(batch['prompt'], truncation=True)


def prepare_df(dataset):
    data_dict = []
    for i, line in enumerate(tqdm(dataset['train'])):
        if i == 1800000:
            break
        else:
            data_dict.append(
                {'prompt': line['question'][2:-3] + " --> " + line['answer'][2:-3]}
            )
    return data_dict


def train(model, criterion, optimizer, train_loader, epochs):
    train_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            targets = batch['input_ids'].clone().detach()
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id

            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs.transpose(2, 1), targets)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_losses[it] = train_loss
        dt = datetime.now() - t0
        print(f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}")

    return train_losses


if __name__ == "__main__":
    res_df = prepare_df(dataset)
    dataset = Dataset.from_list(res_df)
    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ['prompt']
    )
    train_loader = DataLoader(
        tokenized_datasets,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator
    )
    model = Decoder(
                    vocab_size=tokenizer.vocab_size,
                    # vocab_size=64,
                    max_len=tokenizer.max_model_input_sizes[checkpoint],
                    # max_len=64,
                    d_k=16,
                    d_model=64,
                    n_heads=3,
                    n_layers=2,
                    dropout_prob=0.1)

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = train(
        model, criterion, optimizer, train_loader, epochs=10
    )

    torch.save(model.state_dict(), 'models/model.pt')
