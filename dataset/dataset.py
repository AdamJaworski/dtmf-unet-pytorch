from typing import Optional
import numpy as np
import numpy.random as rand
import random
import torch
from torch.utils.data import Dataset
import sounddevice as sd

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
    10: (941, 1209),  # '*'
    11: (941, 1477),  # '#'
    # 12 will represent noise
}

forbidden_freq = {697, 770, 852, 941, 1209, 1336, 1477}

fs = 44100  # Hz

def add_awgn_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

class DTMFDataset(Dataset):
    def __init__(self, datasize: Optional[int]=None, test=False):
        if not test:
            assert datasize, 'datasize is required'

        self.data_size = int(datasize) if not test else 1
        self.data = []
        self.labels = []

        window_duration = 0.12 # in s

        for _ in range(self.data_size):
            digits = []
            # Create 30 digits, each with a random duration between 300ms and 500ms
            for i in range(30):
                new_digit = rand.randint(0, 11), rand.randint(600,900)
                digits.append(new_digit)

            # Calculate total duration (in ms)
            full_sound_duration = 0
            for digit in digits:
                full_sound_duration += window_duration * 1e3   # 150ms break before the digit tone
                full_sound_duration += digit[1]  # digit duration in ms

            # Create time array for the entire signal
            t = np.linspace(0, full_sound_duration / 1000, int(fs * full_sound_duration / 1000), endpoint=False)
            full_long_sound = np.zeros_like(t)

            # Add background noise frequency (not one of the DTMF freqs)
            #for i in range(rand.randint(1,2)):

            background_noise_freq = self.noise_bg = random.choice([rand.randint(100, 400), rand.randint(1700, 2500)])
            while background_noise_freq in forbidden_freq:
                background_noise_freq = rand.randint(100, 2000)

            # Fill with background noise (sine wave + AWGN)
            full_long_sound += np.sin(2 * np.pi * background_noise_freq * t)


            # Ground truth labels: start as noise (12)
            gt_labels = np.full(len(t), 12, dtype=int)

            indicator = 0
            # Place tones
            for digit in digits:
                # Add break
                break_samples = int(window_duration * fs)
                tone_samples = int(digit[1]/1000 * fs) # digit duration

                # The current segment for break: [indicator, indicator+break_samples)
                # Already labeled as 12, so no need to change label for break section:
                # indicator -> indicator + break_samples
                start_break = indicator
                end_break = indicator + break_samples
                # break region should remain label=12, no tone
                # full_long_sound here is already noise+AWGN, no need to overwrite for silence
                indicator = end_break

                # Now the tone segment: [indicator, indicator+tone_samples)
                start_tone = indicator
                end_tone = indicator + tone_samples

                # Get the corresponding DTMF frequencies
                low_freq, high_freq = digit_to_freq[digit[0]]

                t_tone = np.linspace(0, digit[1]/1000, tone_samples, endpoint=False)
                # Create DTMF tone: sum of two sines
                sound_to_append = (np.sin(2 * np.pi * low_freq * t_tone) +
                                   np.sin(2 * np.pi * high_freq * t_tone)) / 2.0
                # Random amplitude scaling
                sound_to_append *= rand.randint(1,10)/10

                # Place tone in full_long_sound
                full_long_sound[start_tone:end_tone] += sound_to_append[:end_tone - start_tone]

                # Set labels for this tone segment
                gt_labels[start_tone:end_tone] = digit[0]

                indicator = end_tone

            # Now split into 100ms windows
            # Each window is 0.1s. Number of windows = total_duration_in_seconds / 0.1
            total_duration_s = len(full_long_sound) / fs
            num_windows = int(total_duration_s / 0.1)  # total_duration_s * 10

            self.snr = rand.randint(20, 100) / 10
            full_long_sound = add_awgn_noise(full_long_sound, self.snr)  # Add AWGN noise

            self.long_sound = full_long_sound
            for window_idx in range(num_windows):
                start = int(window_idx * 0.1 * fs)
                end = int((window_idx + 1) * 0.1 * fs)
                signal = full_long_sound[start:end]
                signal_tensor = torch.from_numpy(signal).float().to(device)

                true_label_long = gt_labels[start:end]
                count = np.bincount(true_label_long, minlength=13)
                true_label = np.argmax(count).tolist()

                self.data.append(signal_tensor)
                self.labels.append(true_label)

            # Update data_size to reflect the total number of windows created
            self.data_size = len(self.labels)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def play_long_sound(self):
        sd.play(self.long_sound, fs)
        sd.wait()

if __name__ == "__main__":
    dataset = DTMFDataset(datasize=5)
    print(dataset.data_size)
    print(dataset.labels)
    dataset.play_long_sound()
