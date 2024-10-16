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
        self.data_size = datasize
        self.data = []

        silence_break = np.zeros(int(fs * 40 / 1000), dtype=np.float32)
        silence_label = np.full(len(silence_break), -1, dtype=np.int64)
        for i in range(self.data_size):
            duration_of_sound = rand.randint(3, 6)  # Number of digits
            sound = []
            labels = []
            for part in range(duration_of_sound):
                digit = rand.randint(0, 11)  # >11 is noise
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

                # Create label array for the signal
                signal_length = len(signal)
                if digit <= 11:
                    label_array = np.full(signal_length, digit, dtype=np.int64)
                else:
                    label_array = np.full(signal_length, -1, dtype=np.int64)

                # Append signal and labels
                sound.append(signal)
                labels.append(label_array)

                # Append silence break and silence label
                sound.append(silence_break)
                labels.append(silence_label)

            full_signal = np.concatenate(sound).astype(np.float32)
            full_labels = np.concatenate(labels).astype(np.int64)
            self.data.append((torch.from_numpy(full_signal).to(gv.device), torch.from_numpy(full_labels).to(gv.device)))

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
