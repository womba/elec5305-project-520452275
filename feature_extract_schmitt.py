import librosa
import matplotlib.pyplot as plt
import numpy as np

filename = "dataset/0.wav"
y, sr = librosa.load(filename, duration=5, sr=48000, mono=True)

D = np.abs(librosa.stft(y))


plt.show()

