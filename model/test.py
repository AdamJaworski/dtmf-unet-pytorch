import os
from sklearn.metrics import f1_score
import settings.global_variables as gv
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.model import Model
from dataset.dataset import DTMFDataset

def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1).long()
    return data_padded, labels_padded

def test():
    model = Model()
    model.to(gv.device)
    model.eval()
    data = DTMFDataset(1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=13)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    state_list = os.listdir(gv.paths.model_path)
    state_list = ['latest.pth']
    for state in state_list:
        model.load_state_dict(torch.load(gv.paths.model_path / state)) #, weights_only=True

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.unsqueeze(1)
                outputs = model(data)
                outputs = outputs.permute(0, 2, 1)
                outputs = outputs.reshape(-1, 14)
                labels = labels.reshape(-1)

                if outputs.size(0) != labels.size(0):
                    min_size = min(outputs.size(0), labels.size(0))
                    outputs = outputs[:min_size]
                    labels = labels[:min_size]

                loss = loss_fn(outputs, labels)
                _, predicted_classes = torch.max(outputs, dim=1)
                predicted_classes = predicted_classes.cpu().numpy()
                labels = labels.cpu().numpy()
                for index, element in enumerate(labels):
                    labels[index] = '_' if element == 13 else element
                for index, element in enumerate(predicted_classes):
                    predicted_classes[index] = '_' if element == 13 else element
                print(f"{state}: (loss: {loss.item()}, gt: {remove_consecutive_duplicates(labels.tolist())}, output: {remove_consecutive_duplicates(predicted_classes.tolist())})")
                score = f1_score(labels.flatten(), predicted_classes.flatten(), average='macro')
                print(f"F1 Score: {score:.4f}")


def remove_similar(list_: list) -> list:
    index = 1
    while index < len(list_):
        len_ = len(list_)
        while list_[index - 1] == list_[index] and index <= len_ - 2:
            del list_[index]
            len_ -= 1
        index += 1
    return list_

def remove_consecutive_duplicates(seq):
    result = []
    prev = None
    for item in seq:
        if item != prev:
            result.append(item)
            prev = item
    return result