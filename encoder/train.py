import configs
from encoder_classifier import Encoder
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from clean_prep import clean_prepare, tokenizer
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

logger = TensorBoardLogger("tb_logs", name="encoder_class")


# classic pytorch training
def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = 0
        n_train = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['input_ids'].size(0)
            n_train += batch['input_ids'].size(0)
        train_loss = train_loss / n_train
        model.eval()
        test_loss = 0
        n_test = 0
        for batch in tqdm(valid_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            test_loss += loss.item() * batch['input_ids'].size(0)
            n_test += batch['input_ids'].size(0)
        test_loss = test_loss / n_test
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        dt = datetime.now() - t0
        print(f"Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}")
    return train_losses, test_losses


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        max_len=configs.MAX_LENGTH,
        d_k=configs.QK_SIZE,
        d_model=configs.EMBED_SIZE,
        n_heads=configs.N_HEADS,
        n_layers=configs.N_LAYERS,
        n_classes=configs.NUM_CLASSES,
        dropout_prob=configs.DROPOUT
    )
    # model.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    # train_loader, valid_loader = clean_prepare()
    # train_losses, test_losses = train(model, criterion, optimizer, train_loader,
    #                                   valid_loader, epochs=configs.EPOCHS)
    #
    # torch.save(model.state_dict(), 'models/model.pt')

    trainer = Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        max_epochs=10,
        min_epochs=3,
        logger=model.logger)

    train_loader, valid_loader = clean_prepare()
    trainer.fit(model, train_loader, valid_loader)
