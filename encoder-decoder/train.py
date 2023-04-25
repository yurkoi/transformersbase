from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from encoder_decoder import Encoder, Decoder, Transformer
from datetime import datetime
import os
import numpy as np
import pandas as pd


dataset = load_dataset("trondizzy/uk_en_combined_OPUS_sets")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_checkpoint = 'Helsinki-NLP/opus-mt-uk-en'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preproc_df(dataset):
    dct_topd = {'en': [], 'uk': []}
    for en, uk in tqdm(zip(dataset['train']['EN'][:1000000],
                           dataset['train']['UK'][:1000000])):
        dct_topd['en'].append(en)
        dct_topd['uk'].append(uk)
    filename = 'en_uk.csv'
    pd.DataFrame(dct_topd).to_csv(filename, index=None)
    raw_dataset = load_dataset('csv', data_files='en_uk.csv')
    split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
    os.system(f'rm -rf {filename}')
    return split


def preprocess_function(batch):
    max_input_length = 256
    max_target_length = 256
    model_inputs = tokenizer(batch['uk'],
                             max_length=max_input_length, truncation=True)
    labels = tokenizer(
        text_target=batch['en'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        n_train = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            enc_input = batch['input_ids']
            enc_mask = batch['attention_mask']
            targets = batch['labels']
            dec_input = targets.clone().detach()
            dec_input = torch.roll(dec_input, shifts=1, dims=1)
            dec_input[:, 0] = 61587
            dec_input = dec_input.masked_fill(
                dec_input == -100, tokenizer.pad_token_id)
            dec_mask = torch.ones_like(dec_input)
            dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)
            outputs = model(enc_input, dec_input, enc_mask, dec_mask)
            loss = criterion(outputs.transpose(2, 1), targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # train_loss += loss.item()*batch['input_ids'].size(0)
            n_train += batch['input_ids'].size(0)
            if n_train % 5000 == 0:
                print(loss)
        train_loss = np.mean(train_loss)
        model.eval()
        test_loss = []
        # n_test = 0
        for batch in tqdm(valid_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            enc_input = batch['input_ids']
            enc_mask = batch['attention_mask']
            targets = batch['labels']
            dec_input = targets.clone().detach()
            dec_input = torch.roll(dec_input, shifts=1, dims=1)
            dec_input[:, 0] = 61587
            dec_input = dec_input.masked_fill(
                dec_input == -100, tokenizer.pad_token_id)
            dec_mask = torch.ones_like(dec_input)
            dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)
            outputs = model(enc_input, dec_input, enc_mask, dec_mask)
            loss = criterion(outputs.transpose(2, 1), targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        dt = datetime.now() - t0
        print(f"\n Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Durtion: {dt}")

    return train_losses, test_losses


if __name__ == "__main__":
    split = preproc_df(dataset)
    tokenized_datasets = split.map(
        preprocess_function,
        batched=True,
        remove_columns=split['train'].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    train_loader = DataLoader(
        tokenized_datasets['train'],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator)
    valid_loader = DataLoader(
        tokenized_datasets['test'],
        # shuffle=True,
        batch_size=8,
        collate_fn=data_collator)
    tokenizer.add_special_tokens({"cls_token": "<s>"})
    encoder = Encoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=512,
                      d_k=16,
                      d_model=128,
                      n_heads=4,
                      n_layers=2,
                      dropout_prob=0.1)
    decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=512,
                      d_k=16,
                      d_model=128,
                      n_heads=4,
                      n_layers=2,
                      dropout_prob=0.1)
    transformer = Transformer(encoder, decoder)
    print(device)
    encoder.to(device)
    decoder.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(transformer.parameters())
    train_losses, test_losses = train(
        transformer, criterion, optimizer, train_loader, valid_loader, epochs=10)
    torch.save(transformer.state_dict(), 'models/model.pt')
