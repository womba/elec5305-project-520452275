import librosa
import matplotlib.pyplot as plt
import numpy as np
import random

def open(filename):
    y, sr = librosa.load(filename, sr=48000, mono=True)

def create_mel(y, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def time_shift(sample, shift_limit):
    sig_len = len(sample)
    shift_amount = int(random.random() * shift_limit * sig_len)
    new_sig = np.roll(sample, shift_amount)
    return new_sig

def freq_mask(sample, max_mask_pct=0.1, n_freq_masks=1):
    aug_sample = np.copy(sample)
    n_mels, _ = sample.shape
    mask_value = np.mean(sample)

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        aug_sample[f0:f0 + f, :] = mask_value

    return aug_sample

def time_mask(sample, max_mask_pct=0.1, n_time_masks=1):
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
    # filename = "dataset/0.wav"
    filename = librosa.ex('trumpet')
    y, sr = librosa.load(filename, duration=2, sr=48000, mono=True)

    plt.figure(figsize=(10, 4))

    librosa.display.waveshow(y, sr=sr)

    # Create Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    aug_spec = time_mask(mel_spectrogram_db)
    aug_spec = time_shift(aug_spec, 20)


    fig, ax = plt.subplots()

    img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Mel-frequency spectrogram')

    plt.show()

if __name__ == "__main__":
    main()
