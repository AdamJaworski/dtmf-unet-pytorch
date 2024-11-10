import math
import sys

import numpy as np
import numpy.random as rand
import torch
from torch.utils.data import Dataset
from settings import global_variables as gv

digit_to_freq = {
    1: (697, 1209),
    2: (697, 1336),
    3: (697, 1477),
    4: (770, 1209),
    5: (770, 1336),
    6: (770, 1477),
    7: (852, 1209),
    8: (852, 1336),
    9: (852, 1477),
    10: (941, 1209),
    0: (941, 1336),
    11: (941, 1477)
}

fs = 44100  # Hz

class DTMFDataset(Dataset):
    def __init__(self, datasize):
        self.data_size = int(datasize)
        self.data = []
        frame_size = 275  # Samples per label

        silence_break = np.zeros(int(fs * 40 / 1000), dtype=np.float32)
        for i in range(self.data_size):
            duration_of_sound = rand.randint(3, 6)  # Number of digits
            sound = []
            labels = []
            digits = []  # Store digits used
            signal_lengths = []  # Store signal lengths
            silence_lengths = []  # Store silence lengths
            for part in range(duration_of_sound):
                digit = rand.randint(0, 11)  # >11 is noise
                digits.append(digit)
                duration = rand.randint(100, 300)  # ms
                snr = rand.randint(1, 15)
                volume = rand.uniform(0.5, 1.0)

                duration_samples = int(fs * duration / 1000)
                t = np.arange(duration_samples) / fs

                if digit > 11:
                    sin1 = np.sin(2 * np.pi * rand.randint(1800, 15000) * t) * volume
                    sin2 = np.sin(2 * np.pi * rand.randint(1800, 15000) * t) * volume
                else:
                    freqs = digit_to_freq[digit]
                    sin1 = np.sin(2 * np.pi * freqs[0] * t) * volume
                    sin2 = np.sin(2 * np.pi * freqs[1] * t) * volume

                signal = sin1 + sin2
                signal = add_awgn_noise(signal, snr)
                signal = np.clip(signal, -1.0, 1.0)

                # Append signal and its length
                sound.append(signal)
                signal_lengths.append(len(signal))

                # Append silence break and its length
                sound.append(silence_break)
                silence_lengths.append(len(silence_break))

            # Concatenate all sound segments
            full_signal = np.concatenate(sound).astype(np.float32)

            # Add 1 second of silence at the end
            end_silence_duration = fs  # 1 second of silence
            end_silence = np.zeros(end_silence_duration, dtype=np.float32)
            full_signal = np.concatenate([full_signal, end_silence]).astype(np.float32)

            # Now, compute the labels for the entire full_signal
            total_frames = math.ceil(len(full_signal) / frame_size)
            labels = []

            for idx, (digit, signal_len, silence_len) in enumerate(zip(digits, signal_lengths, silence_lengths)):
                # Compute labels for signal
                num_signal_frames = math.ceil(signal_len / frame_size)
                label_value = digit if digit <= 11 else 12
                signal_labels = np.full(num_signal_frames, label_value, dtype=np.int64)

                # Compute labels for silence
                num_silence_frames = math.ceil(silence_len / frame_size)
                silence_labels = np.full(num_silence_frames, 13, dtype=np.int64)

                # Append labels
                labels.extend(signal_labels)
                labels.extend(silence_labels)

            # Compute labels for the end silence
            end_silence_frames = math.ceil(len(end_silence) / frame_size)
            end_silence_labels = np.full(end_silence_frames, 13, dtype=np.int64)
            labels.extend(end_silence_labels)

            full_labels = np.array(labels, dtype=np.int64)
            # Ensure the labels match the number of frames
            # assert len(full_labels) == total_frames, f"Mismatch between labels ({len(full_labels)}) and frames ({total_frames})"

            # print(len(full_signal), len(full_labels))
            self.data.append((
                torch.from_numpy(full_signal).to(gv.device),
                torch.from_numpy(full_labels).to(gv.device)
            ))

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index]

def add_awgn_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
