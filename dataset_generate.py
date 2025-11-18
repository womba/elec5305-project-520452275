'''
This program converts the keybaord audio samples provided in /dataset into (x) individual samples labelled (x)_1, (x)_2 and so forth.
This is achieved by utilising librosa's in-built onset_function which detects peaks by thresholding it's normalised power and backtracks
these peaks to the previous local minima. We then take a 0.33s snapshot from that point and define that to be our sample. We repeat this process
for every .wav file located in /dataset and save the resultant waveforms in /output.

'''

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

path = "dataset/Testing"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for file in files:
    filename = f"dataset/Testing/{file}"
    y, sr = librosa.load(filename, sr=48000, mono=True)

    D = np.abs(librosa.stft(y))

    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_envelope, sr=sr)

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, hop_length=512, backtrack=True, sr=sr, wait=20)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)

    for i in range(len(onset_samples)-1):
        onset_samples[i] += 48000/3
        post_onset_frames = librosa.samples_to_frames(onset_samples, hop_length=512)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0], sr=sr)
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    ax[1].plot(onset_times, onset_envelope, label="Onset Strength")
    ax[1].vlines(onset_times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[1].vlines(onset_times[post_onset_frames], 0, onset_envelope.max(), color='g', alpha=0.9, linestyle='--', label='Onsets-Post')
    ax[1].legend()

    fig, ax = plt.subplots(nrows=len(onset_samples)*2, sharex=True)

    plt.show()

    break

    # UNCOMMENT TO GENERATE SAMPLES
    # for i in range(0, len(onset_samples)):
    #     start = librosa.frames_to_samples(onset_frames[i], hop_length=512)
    #     end = librosa.frames_to_samples(post_onset_frames[i], hop_length=512)
    #     sample = y[start:end]

    #     if len(sample) > 0:
    #         rms = np.sqrt(np.mean(sample ** 2))
    #     else:
    #         rms = 0.0
    #     target_rms = 0.1
    #     if rms > 0:
    #         sample = sample * (target_rms / rms)

    #     path = f"dataset/testing/testing_output/{file[0].upper()}_{i}.wav"
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     sf.write(path, sample, 48000, subtype='PCM_24', format='wav')





# filename = "dataset/0.wav"
# y, sr = librosa.load(filename, sr=48000, mono=True)

# D = np.abs(librosa.stft(y))

# onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
# onset_times = librosa.times_like(onset_envelope, sr=sr)

# onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, hop_length=512, backtrack=True, sr=sr, wait=10)
# onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)

# for i in range(len(onset_samples)):
#     onset_samples[i] += 48000/3
# post_onset_frames = librosa.samples_to_frames(onset_samples, hop_length=512)

# fig, ax = plt.subplots(nrows=2, sharex=True)

# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0], sr=sr)
# ax[0].set(title='Power spectrogram')
# ax[0].label_outer()

# ax[1].plot(onset_times, onset_envelope, label="Onset Strength")
# ax[1].vlines(onset_times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
# ax[1].vlines(onset_times[post_onset_frames], 0, onset_envelope.max(), color='g', alpha=0.9, linestyle='--', label='Onsets-Post')
# ax[1].legend()

# fig, ax = plt.subplots(nrows=len(onset_samples)*2, sharex=True)


# for i in range(0, len(onset_samples)):
#     start = librosa.frames_to_samples(onset_frames[i], hop_length=512)
#     end = librosa.frames_to_samples(post_onset_frames[i], hop_length=512)
#     sample = y[start:end]

#     path = f"output/0_{i}.wav"
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     sf.write(path, sample, 48000, subtype='PCM_24', format='wav')

    # # Plot for Visualisation
    # mel_spectrogram = librosa.feature.melspectrogram(y=sample, sr=sr)
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[i*2])
    # ax[i*2].set(title='Mel spectrogram')

    # t_axis = np.arange(len(sample)) / sr
    # ax[i*2+1].plot(t_axis, sample, label="data1")

# plt.show()

# data1start = librosa.frames_to_samples(onset_frames[0], hop_length=512)
# data1end = librosa.frames_to_samples(post_onset_frames[0], hop_length=512)
# data1_samples = y[data1start:data1end]

# fig2, ax2 = plt.subplots(nrows=2, sharex=True)

# mel_spectrogram = librosa.feature.melspectrogram(y=data1_samples, sr=sr)
# mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax2[0])
# ax2[0].set(title='Mel spectrogram')

# time1 = np.arange(len(data1_samples)) / sr
# ax2[1].plot(time1, data1_samples, label="data1")


