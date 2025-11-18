import librosa
import matplotlib.pyplot as plt
import numpy as np
import random

def open(filename):
    y, sr = librosa.load(filename, sr=48000, mono=True)

def create_mel(y, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=2048, win_length=4096)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def create_linear_spectrogram(y, sr):
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=512, win_length=512))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    return stft_db


def time_shift(sample, shift_limit):
    sig_len = len(sample)
    shift_amount = int(random.random() * shift_limit * sig_len)
    new_sig = np.roll(sample, shift_amount)
    return new_sig

def freq_mask(sample, max_mask_pct=0.2, n_freq_masks=1):
    aug_sample = np.copy(sample)
    n_mels, _ = sample.shape
    mask_value = np.mean(sample)

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        aug_sample[f0:f0 + f, :] = mask_value

    return aug_sample

def time_mask(sample, max_mask_pct=0.2, n_time_masks=1):
    aug_sample = np.copy(sample)
    _ , n_steps = sample.shape
    mask_value = np.mean(sample)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, n_steps - time_mask_param)
        aug_sample[:, t0:t0 + t] = mask_value

    return aug_sample

def main():
    # Load File + Resample to 48kHz + Convert to Mono
    filename = "output/A_3.wav"
    # filename = librosa.ex('trumpet')
    y, sr = librosa.load(filename, duration=2, sr=48000, mono=True)

    # Create Linear Spectrogram
    linear_spec = create_linear_spectrogram(y, sr)

    # Create Mel-Spectrogram with augmentations
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    aug_spec = time_mask(mel_spectrogram_db, max_mask_pct=0.3, n_time_masks=1)
    # aug_spec = time_shift(aug_spec, 20)
    aug_spec = freq_mask(aug_spec, max_mask_pct=0.3, n_freq_masks=1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Plot Waveform
    ax1 = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')

    # 2. Plot Linear Spectrogram
    ax2 = axes[1]
    img2 = librosa.display.specshow(linear_spec, x_axis='time', y_axis='hz', sr=sr, ax=ax2)
    ax2.set_title('Linear Spectrogram')
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')

    # 3. Plot Mel Spectrogram with Augmentations
    ax3 = axes[2]
    img3 = librosa.display.specshow(aug_spec, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax3)
    ax3.set_title('Mel Spectrogram with Augmentations')
    plt.colorbar(img3, ax=ax3, format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
