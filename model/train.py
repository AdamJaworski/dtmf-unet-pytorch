import pathlib
import time
import settings.global_variables as gv
import torch.nn as nn
import torch
from torch.optim import Adam
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
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1).long()
    return data_padded, labels_padded

def train():
    num_classes = 12
    model = Model(num_classes=num_classes)
    if pathlib.Path.exists(gv.paths.model_path / 'latest.pth'):
        print("Loading saved state")
        model.load_state_dict(torch.load(gv.paths.model_path / 'latest.pth', weights_only=True))
    model.to(gv.device)
    optimizer = Adam(params=model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10000)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    print("Creating dataset...")
    dataset_ = DTMFDataset(3e3)
    dataloader = DataLoader(dataset_, batch_size=16, shuffle=True, collate_fn=collate_fn)

    num_epochs = 10000

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}')
        total_loss = 0.0
        for data, labels in dataloader:
            data = data.unsqueeze(1)  # Add channel dimension
            labels = labels.long()  # Ensure labels are LongTensor
            optimizer.zero_grad()

            outputs = model(data)
            outputs = outputs.permute(0, 2, 1)
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)

            if outputs.size(0) != labels.size(0):
                min_size = min(outputs.size(0), labels.size(0))
                outputs = outputs[:min_size]
                labels = labels[:min_size]

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            scheduler.step(loss.item())

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')
        save_model(model)

def save_model(model_instance):
    model_instance.to('cpu')
    save(model_instance.state_dict(),gv.paths.model_path / f'{time.time()}.pth')
    if pathlib.Path.exists(gv.paths.model_path / 'latest.pth'):
        pathlib.Path.unlink(gv.paths.model_path / 'latest.pth')
    save(model_instance.state_dict(), gv.paths.model_path / 'latest.pth')
    model_instance.to(gv.device)

if __name__ == "__main__":
    train()
