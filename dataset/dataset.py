from typing import Optional
import numpy as np
import numpy.random as rand
import random
import torch
from torch.utils.data import Dataset
import sounddevice as sd
import os

try:
    import librosa

    AUDIO_LOADER = 'librosa'
except ImportError:
    # fallback if librosa is not installed
    import soundfile as sf

    AUDIO_LOADER = 'soundfile'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

digit_to_freq = {
    0: (941, 1336),  # Digit '0'
    1: (697, 1209),
    2: (697, 1336),  # Digit '2'
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

fs = 44100  # sample rate


def add_awgn_noise(signal, snr_db):
    """
    Adds white Gaussian noise to a signal at a specified SNR (dB).
    """
    signal_power = np.mean(signal ** 2) if np.mean(signal ** 2) != 0 else 1e-12
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def load_random_noise_file(fs=44100, folder='.', prefix='sample-', ext='.wav'):
    """
    Loads a random noise audio file from a given folder (matching prefix and extension).
    Returns the loaded noise as a numpy array (resampled to fs if necessary).
    """
    # Collect all files that match prefix and ext
    all_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]

    if not all_files:
        # No files found, return zeros as fallback
        return np.zeros(fs)

    # Choose a random file
    noise_file = random.choice(all_files)

    file_path = os.path.join(folder, noise_file)

    if AUDIO_LOADER == 'librosa':
        noise_audio, sr = librosa.load(file_path, sr=fs, mono=True)
    else:
        noise_audio, sr = sf.read(file_path)
        # Resample if needed
        if sr != fs:
            noise_audio = librosa.resample(noise_audio, orig_sr=sr, target_sr=fs)

    return noise_audio / np.std(noise_audio)


def apply_impulse_noise(signal, num_spikes=5, max_amplitude=0.5):
    """
    Inserts random impulses (spikes) into the signal.
    num_spikes: number of random spikes
    max_amplitude: max amplitude for spike
    """
    length = len(signal)
    for _ in range(num_spikes):
        idx = np.random.randint(0, length)
        amp = np.random.uniform(-max_amplitude, max_amplitude)
        signal[idx] += amp
    return signal


def fade_in_out(tone, fade_duration=0.01, fs=44100):
    """
    Applies a linear fade-in and fade-out to a tone.
    fade_duration is in seconds.
    """
    fade_samples = int(fade_duration * fs)
    length = len(tone)
    if fade_samples == 0 or fade_samples * 2 > length:
        return tone  # not enough samples to fade

    # Fade-in
    for i in range(fade_samples):
        tone[i] *= (i / fade_samples)
    # Fade-out
    for i in range(fade_samples):
        tone[length - 1 - i] *= (i / fade_samples)
    return tone


class DTMFDataset(Dataset):
    def __init__(self, datasize: Optional[int] = None, test=False, noise_folder='.',
                 use_noise_files=True):
        """
        :param datasize: Number of 'scenarios' to create if not in test mode.
        :param test: If True, only create a single scenario.
        :param noise_folder: Path to the folder containing sample-X.webm noise files.
        :param use_noise_files: Whether to use external noise files in the mix.
        """
        if not test:
            assert datasize, 'datasize is required'

        self.data_size = int(datasize) if not test else 1
        self.data = []
        self.labels = []

        # We'll generate one "scenario" at a time; each scenario has multiple 100ms windows
        for scenario_idx in range(self.data_size):

            # Generate 30 digits, each with random durations between 300ms and 600ms
            # Also generate random break durations between 100ms and 300ms
            digits = []
            for i in range(30):
                tone_duration = rand.randint(300, 600)  # ms
                break_duration = rand.randint(200, 600)  # ms
                digit_index = rand.randint(0, 12)  # 0-11 are digits, 12 = noise
                digits.append((digit_index, tone_duration, break_duration))

            # Calculate total duration (in ms) for this scenario
            full_sound_duration_ms = 0
            for d in digits:
                full_sound_duration_ms += d[2]  # break
                full_sound_duration_ms += d[1]  # tone

            # Create time array
            t = np.linspace(
                0,
                full_sound_duration_ms / 1000,
                int(fs * (full_sound_duration_ms / 1000)),
                endpoint=False
            )
            full_long_sound = np.zeros_like(t)

            # Ground truth labels: start as noise (12)
            gt_labels = np.full(len(t), 12, dtype=int)

            # Background noise frequency (still keep old approach for a single background tone)
            background_noise_freq = random.choice([rand.randint(100, 400), rand.randint(1700, 2500)])
            while background_noise_freq in forbidden_freq:
                background_noise_freq = rand.randint(100, 2000)
            # Add background sine wave
            full_long_sound += np.sin(2 * np.pi * background_noise_freq * t)

            # Optionally add external noise from .webm files (e.g. real-world recordings)
            if use_noise_files:
                external_noise = load_random_noise_file(fs=fs, folder=noise_folder, prefix='sample-', ext='.wav')
                # If shorter, tile it or if longer, trim it
                if len(external_noise) < len(full_long_sound):
                    # Tile up to match length
                    repeats = (len(full_long_sound) // len(external_noise)) + 1
                    external_noise = np.tile(external_noise, repeats)
                external_noise = external_noise[:len(full_long_sound)]
                # Mix in external noise at some random amplitude
                noise_gain = np.random.uniform(0.1, 0.5)
                full_long_sound += noise_gain * external_noise

            # Now place tones
            indicator = 0
            for digit, tone_duration, break_duration in digits:
                # break
                break_samples = int(break_duration * fs / 1000)
                tone_samples = int(tone_duration * fs / 1000)

                # Next region: break => label=12, do nothing (already 12)
                indicator += break_samples

                # Now tone region
                start_tone = indicator
                end_tone = indicator + tone_samples
                if digit < 12:
                    # DTMF freq (with slight randomization ±1-3 Hz)
                    low_freq, high_freq = digit_to_freq[digit]
                    low_freq_variation = random.uniform(-3, 3)
                    high_freq_variation = random.uniform(-3, 3)
                    low_freq += low_freq_variation
                    high_freq += high_freq_variation

                    # Time array for just this tone
                    t_tone = np.linspace(0, tone_duration / 1000, tone_samples, endpoint=False)
                    # Create tone
                    tone_signal = 0.5 * (np.sin(2 * np.pi * low_freq * t_tone) +
                                         np.sin(2 * np.pi * high_freq * t_tone))

                    # Random amplitude scaling
                    scale = rand.uniform(0.5, 1.0)
                    tone_signal *= scale

                    # Apply fade in/out
                    tone_signal = fade_in_out(tone_signal, fade_duration=0.01, fs=fs)
                    #tone_signal = add_awgn_noise(tone_signal, np.random.uniform(15, 25))

                    # Place tone in full_long_sound
                    full_long_sound[start_tone:end_tone] += tone_signal[:end_tone - start_tone]

                    # Set labels for this tone region
                    gt_labels[start_tone:end_tone] = digit
                # else digit==12 => just random noise, we do nothing special
                indicator += tone_samples

            # Optionally apply random impulse noise throughout full signal
            full_long_sound = apply_impulse_noise(full_long_sound, num_spikes=rand.randint(2, 8),
                                                  max_amplitude=0.3)

            # Instead of one single SNR, let's vary SNR in 100ms segments over the entire signal
            # We'll do short-time segments and apply different SNR in each
            segment_size = int(0.1 * fs)
            for seg_start in range(0, len(full_long_sound), segment_size):
                seg_end = min(seg_start + segment_size, len(full_long_sound))
                # random SNR between 5 dB and 20 dB
                random_snr = np.random.uniform(5, 20)
                full_long_sound[seg_start:seg_end] = add_awgn_noise(full_long_sound[seg_start:seg_end], random_snr)

            #full_long_sound = add_awgn_noise(full_long_sound, np.random.uniform(25, 35))
            full_long_sound = full_long_sound * 0.05
            self.long_sound = full_long_sound

            # Now split final signal into 100ms windows, apply Hamming window,
            # pick the majority label
            num_windows = int(len(full_long_sound) / segment_size)
            for window_idx in range(num_windows):
                start = window_idx * segment_size
                end = (window_idx + 1) * segment_size
                signal = full_long_sound[start:end]

                # Apply hamming window
                windowed_signal = signal * np.hamming(len(signal))

                signal_tensor = torch.from_numpy(windowed_signal).float().to(device)
                true_label_long = gt_labels[start:end]
                # majority label
                count = np.bincount(true_label_long, minlength=13)
                true_label = np.argmax(count).tolist()

                self.data.append(signal_tensor)
                self.labels.append(true_label)

        # Update data_size to reflect total # of 100ms windows
        self.data_size = len(self.labels)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def play_long_sound(self):
        sd.play(self.long_sound, fs)
        sd.wait()

    def save(self):
        from scipy.io.wavfile import write
        write('dataset_fixed3.wav', fs, self.long_sound)


if __name__ == "__main__":
    dataset = DTMFDataset(datasize=1, noise_folder='.', use_noise_files=True)
    print(dataset.data_size)
    print(dataset.labels)
    dataset.save()

