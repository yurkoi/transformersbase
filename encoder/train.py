from encoder_classifier import Encoder
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from clean_prep import clean_prepare, tokenizer
import torch.nn as nn
from torchmetrics.classification import Accuracy, F1Score
import wandb
from configs import config


def arg_parser():
    parser = argparse.ArgumentParser(description='Encoder training args')
    parser.add_argument(
        '--n_layers',
        type=int,
        default=config['n_layers'],
        help='number of transformer blocks'
    )
    parser.add_argument(
        '--classes',
        type=int,
        default=config['classes'],
        help='num of classes')
    parser.add_argument(
        '--max_length',
        type=int,
        default=config['max_length'],
        help='')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=config['vocab_size'],
        help='vocabulary size')
    parser.add_argument(
        '--embed_size',
        type=int,
        default=config['embed_size'],
        help='embeddings size')
    parser.add_argument(
        '--n_heads',
        type=int,
        default=config['n_heads'],
        help='number of heads for q,k,v')
    parser.add_argument(
        '--qk_size',
        type=int,
        default=config['qk_size'],
        help='query-key linear units')
    parser.add_argument(
        '--dropout',
        type=float,
        default=config['dropout'],
        help='dropout')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=config['learning_rate'],
        help='LR')
    parser.add_argument(
        '--epochs',
        type=int,
        default=config['epochs'],
        help='number of epochs for training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config['batch_size'],
        help='')
    parser.add_argument(
        '--wandb_key',
        type=str,
        default="924e8f26e3301fe99fb2d8a673f5947747e3dce8",
        help='wandb experiments tracking key')
    return parser


def make(configs):
    train_loader, valid_loader = clean_prepare()
    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        max_len=configs.max_length,
        d_k=configs.qk_size,
        d_model=configs.embed_size,
        n_heads=configs.n_heads,
        n_layers=configs.n_layers,
        n_classes=configs.classes,
        dropout_prob=configs.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return model, train_loader, valid_loader, criterion, optimizer


def pipeline(parameters):
    with wandb.init(config=parameters):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        for it in range(config.epochs):
            train(model, train_loader, criterion, optimizer, it)
            test(model, test_loader, criterion, it)
    return model


def train(model, train_loader, criterion, optimizer, epoch):
    wandb.watch(model, criterion, log="all", log_freq=10)
    model.train()
    t0 = datetime.now()
    train_loss = 0
    train_acc = 0
    train_f1 = 0
    n_train = 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        acc = accuracy(outputs, batch['labels'])
        f1 = f1met(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch['input_ids'].size(0)
        train_acc += acc.item() * batch['input_ids'].size(0)
        train_f1 += f1.item() * batch['input_ids'].size(0)
        n_train += batch['input_ids'].size(0)
    train_loss = train_loss / n_train
    train_acc = train_acc / n_train
    train_f1 = train_f1 / n_train
    dt = datetime.now() - t0
    log = {"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_acc, "Train f1": train_f1}
    print(f"Duration: {dt}")
    print(log)
    wandb.log(log)


def test(model, valid_loader, criterion,  epoch):
    model.eval()
    t0 = datetime.now()
    test_loss = 0
    test_acc = 0
    test_f1 = 0
    n_test = 0
    for batch in tqdm(valid_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        acc = accuracy(outputs, batch['labels'])
        f1 = f1met(outputs, batch['labels'])
        test_loss += loss.item() * batch['input_ids'].size(0)
        test_acc += acc.item() * batch['input_ids'].size(0)
        test_f1 += f1.item() * batch['input_ids'].size(0)
        n_test += batch['input_ids'].size(0)
    test_loss = test_loss / n_test
    test_acc = test_acc / n_test
    test_f1 = test_f1 / n_test
    dt = datetime.now() - t0
    log = {"Epoch": epoch, "Test loss": test_loss, "Test Accuracy": test_acc, "Test f1": test_f1}
    print(f"Duration: {dt}")
    print(log)
    wandb.log(log)
    wandb.save("model.pt")


if __name__ == "__main__":
    parser = arg_parser().parse_args()

    wandb.login(key=parser.wandb_key)
    wandb.init(config=parser)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    f1met = F1Score(task="multiclass", num_classes=3).to(device)
    model = pipeline(parser)


