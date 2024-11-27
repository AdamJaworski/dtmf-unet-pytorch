import os
import sys

from sklearn.metrics import f1_score
import settings.global_variables as gv
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.model import Model
from dataset.dataset import DTMFDataset
import numpy as np


map_keys = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '*',
    11: '#',
    12: '',
    13: ''
}

def test(latest: bool):
    model = Model()
    model.to(gv.device)
    model.eval()
    input_size = 1e3
    data = DTMFDataset(input_size)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    state_list = ['latest.pth'] if latest else os.listdir(gv.paths.model_path)

    highest_f1 = 0
    state_hi = ''
    for state in state_list:
        if len(state_list) > 1:
            print(f'{state}:')
        model.load_state_dict(torch.load(gv.paths.model_path / state, weights_only=True)) #

        with torch.no_grad():
            y_true = []
            y_pred = []
            total_loss = 0
            for data, labels in dataloader:
                data = data.unsqueeze(1)
                outputs = model(data)

                loss = loss_fn(outputs, labels)
                _, predicted_classes = torch.max(outputs, dim=1)
                predicted_classes = predicted_classes.cpu().numpy()
                labels = labels.cpu().numpy()

                y_pred.extend(predicted_classes)
                y_true.extend(labels)

                total_loss += loss.item()


        # Convert lists to tensors for calculation
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)

        # Calculating precision, recall, and F1 score using PyTorch
        TP = 0
        N  = 0
        for index in range(len(y_true_tensor)):
            if y_true_tensor[index] == y_pred_tensor[index]:
                TP += 1
            else:
                N += 1
        precision = TP / (TP + N) if TP + N > 0 else 0
        recall = TP / (TP + N) if TP + N > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > highest_f1:
            highest_f1 = f1
            state_hi = state

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Loss: {total_loss/input_size}\n')

    print(f'Highest F1: {highest_f1}')
    print(f'State: {state_hi}')

def test2():
    import librosa

    model = Model()
    model.eval()
    window_length = 200  # in milliseconds
    total_samples = int(44100 * window_length / 1000)
    model.load_state_dict(torch.load(gv.paths.model_path / 'best.pth'))

    # Load the audio data
    data, sr = librosa.load(gv.paths.chal_path, sr=44100, mono=False)
    print(f"Loaded audio data with {len(data)} samples at {sr} Hz")

    output_total = []
    num_windows = int(len(data) / total_samples)
    print(f"Processing {num_windows} windows of size {total_samples} samples")

    for i in range(num_windows):
        start_idx = i * total_samples
        end_idx = start_idx + total_samples
        input_segment = data[start_idx:end_idx]
        input_segment /= np.max(input_segment)

        # if len(input_segment) < total_samples:
        #     # Pad with zeros if necessary
        #     input_segment = np.pad(input_segment, (0, total_samples - len(input_segment)), 'constant')

        input_segment = torch.from_numpy(input_segment).unsqueeze(0).unsqueeze(1)

        input_segment = input_segment.to(gv.device)

        with torch.no_grad():
            output = model(input_segment)
            _, predicted_classes = torch.max(output, dim=1)
            output_total.append(predicted_classes.cpu().numpy().tolist()[0])

    print("Raw Output Classes:", output_total)
    output_total = remove_similar(output_total)
    print("Filtered Output Classes:", output_total)

    mapped_output = []
    for digit in output_total:
        mapped_output.append(map_keys[digit])

    print(''.join(mapped_output))

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

if __name__ == "__main__":
    test2()