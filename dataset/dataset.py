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
    # 13 is silent
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
        self.labels = []

        for _ in range(self.data_size):
            window_length = 200  # Total window length in milliseconds
            digit = rand.randint(0, 14)  # Random digit between 0 and 13
            length_of_sound = rand.randint(50, window_length + 1)  # Sound duration between 50ms and 200ms
            length_of_silent_total = window_length - length_of_sound  # Total silence duration
            length_of_silent_front = rand.randint(0, length_of_silent_total + 1)  # Silence at the front


            # Convert durations from milliseconds to sample counts
            total_samples = int(fs * window_length / 1000)
            sound_samples = int(fs * length_of_sound / 1000)
            silent_front_samples = int(fs * length_of_silent_front / 1000)
            silent_back_samples = total_samples - sound_samples - silent_front_samples

            if digit < 12:
                freq = digit_to_freq[digit]

                # Generate time array for the sound duration
                t = np.arange(sound_samples) / fs

                # Generate the two sine waves for DTMF
                sin1 = np.sin(2 * np.pi * freq[0] * t)
                sin2 = np.sin(2 * np.pi * freq[1] * t)

                sound = sin1 + sin2

                sound *= rand.randint(1, 11) / 10 # amplituda od 0.1 do 1

                for i in range(rand.randint(3, 35)):
                    noise = rand.randint(1500, 15000)
                    sound += np.sin(2 * np.pi * noise * t)

                # Optionally add noise to the sound
                snr_db = rand.uniform(0.1, 15)  # Random SNR between 20 and 30 dB
                sound = add_awgn_noise(sound, snr_db)

                # Concatenate silence and sound
                silent_front = np.zeros(silent_front_samples, dtype=np.float32)
                silent_back = np.zeros(silent_back_samples, dtype=np.float32)
                signal = np.concatenate((silent_front, sound, silent_back)).astype(np.float32)

            elif digit == 12:
                # Generate noise
                signal = np.random.normal(0, 1, total_samples).astype(np.float32)
                t = np.arange(total_samples) / fs
                for i in range(rand.randint(3, 35)):
                    noise = rand.randint(1500, 15000)
                    signal += np.sin(2 * np.pi * noise * t)

            elif digit == 13:
                # Generate silence
                signal = np.zeros(total_samples, dtype=np.float32)

            # Convert signal to torch tensor
            signal_tensor = torch.from_numpy(signal).float().to(device)

            # Append to data and labels
            self.data.append(signal_tensor)
            self.labels.append(digit)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
