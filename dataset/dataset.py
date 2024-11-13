import math
import numpy as np
import numpy.random as rand
import torch
from torch.utils.data import Dataset

# Define the device (assuming gv.device might not be defined)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

digit_to_freq = {
    0: (941, 1336),   # Digit '0'
    1: (697, 1209),
    2: (697, 1336),
    3: (697, 1477),
    4: (770, 1209),
    5: (770, 1336),
    6: (770, 1477),
    7: (852, 1209),
    8: (852, 1336),
    9: (852, 1477),
    10: (941, 1209),  # '*' key
    11: (941, 1477),  # '#' key
    # 12 will represent noise
}

fs = 44100  # Hz

def add_awgn_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

class DTMFDataset(Dataset):
    def __init__(self, datasize):
        self.data_size = int(datasize)
        self.data = []
        frame_size = 275  # Samples per label

        silence_break = np.zeros(int(fs * 40 / 1000), dtype=np.float32)

        for _ in range(self.data_size):
            duration_of_sound = rand.randint(3, 6)  # Number of digits
            sound = []
            segments = []  # Store (label, length) tuples

            for _ in range(duration_of_sound):
                digit = rand.randint(0, 13)  # 0-11 are digits, 12 is noise
                duration = rand.randint(100, 300)  # ms
                snr = rand.randint(1, 15)
                volume = rand.uniform(0.5, 1.0)

                duration_samples = int(fs * duration / 1000)
                t = np.arange(duration_samples) / fs

                if digit == 12:
                    # Generate noise
                    sin1 = np.sin(2 * np.pi * rand.randint(1800, 15000) * t) * volume
                    sin2 = np.sin(2 * np.pi * rand.randint(1800, 15000) * t) * volume
                else:
                    freqs = digit_to_freq[digit]
                    sin1 = np.sin(2 * np.pi * freqs[0] * t) * volume
                    sin2 = np.sin(2 * np.pi * freqs[1] * t) * volume

                signal = sin1 + sin2
                signal = add_awgn_noise(signal, snr)
                signal = np.clip(signal, -1.0, 1.0)

                # Append signal and its label
                sound.append(signal)
                segments.append((digit, len(signal)))  # digit can be 0-12

                # Append silence break and its label
                sound.append(silence_break)
                segments.append(('silence', len(silence_break)))

            # Concatenate all sound segments
            full_signal = np.concatenate(sound).astype(np.float32)

            # Add 1 second of silence at the end
            end_silence_duration = fs  # 1 second of silence
            end_silence = np.zeros(end_silence_duration, dtype=np.float32)
            full_signal = np.concatenate([full_signal, end_silence]).astype(np.float32)

            # Append end silence to segments
            segments.append(('silence', len(end_silence)))

            # Now, compute the labels for the entire full_signal
            total_frames = math.ceil(len(full_signal) / frame_size)
            labels = []

            for label, segment_len in segments:
                num_frames = math.ceil(segment_len / frame_size)
                if label == 'silence':
                    frame_labels = np.full(num_frames, 13, dtype=np.int64)
                else:
                    # Assign label for digits and noise
                    label_value = label if label <= 11 else 12
                    frame_labels = np.full(num_frames, label_value, dtype=np.int64)
                labels.extend(frame_labels)

            # Adjust labels to match total_frames
            full_labels = np.array(labels, dtype=np.int64)
            if len(full_labels) > total_frames:
                full_labels = full_labels[:total_frames]
            elif len(full_labels) < total_frames:
                # Pad with silence labels
                full_labels = np.pad(full_labels, (0, total_frames - len(full_labels)), 'constant', constant_values=13)

            # Store the data
            self.data.append((
                torch.from_numpy(full_signal).to(device),
                torch.from_numpy(full_labels).to(device)
            ))

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index]
