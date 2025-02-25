import pathlib
import sys
import time
import settings.global_variables as gv
import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.model import Model
from dataset.dataset import DTMFDataset
from torch import save
from torch.optim.lr_scheduler import ReduceLROnPlateau

def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=13).long()
    return data_padded, labels_padded

def train():
    model = Model()
    if pathlib.Path.exists(gv.paths.model_path / 'latest.pth'):
        print("Loading saved state")
        model.load_state_dict(torch.load(gv.paths.model_path / 'latest.pth', weights_only=True)) #
    model.to(gv.device)
    optimizer = AdamW(params=model.parameters(), lr=0.001, weight_decay=1e-5)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_trainable_params)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    loss_fn = nn.CrossEntropyLoss()

    print("Creating dataset...")
    dataset_ = DTMFDataset(128)
    print(f'Size of dataset {len(dataset_)}')
    dataloader = DataLoader(dataset_, batch_size=2048, shuffle=True) #, collate_fn=collate_fn

    num_epochs = 10000

    for epoch in range(1, num_epochs):
        print(f'Starting epoch {epoch}')
        total_loss = 0.0
        for data, labels in dataloader:
            data = data.unsqueeze(1)
            # print(data.size())
            # sys.exit()
            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')
        scheduler.step(average_loss)
        save_model(model, epoch)

def save_model(model_instance, epoch):
    model_instance.to('cpu')
    save(model_instance.state_dict(),gv.paths.model_path / f'{epoch}.pth')
    if pathlib.Path.exists(gv.paths.model_path / 'latest.pth'):
        pathlib.Path.unlink(gv.paths.model_path / 'latest.pth')
    save(model_instance.state_dict(), gv.paths.model_path / 'latest.pth')
    model_instance.to(gv.device)

if __name__ == "__main__":
    train()
