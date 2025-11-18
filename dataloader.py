from torch.utils.data import Dataset
import torch
import librosa
import os
import audio_util
import glob
import matplotlib.pyplot as plt
import numpy as np


class AudioDatasetLoader(Dataset):
  def __init__(self, data_path, indices=None, apply_augmentations=True, class_to_idx=None):
    self.data_path = data_path
    self.audio_files = sorted(glob.glob(os.path.join(data_path, "*.wav")))
    self.duration = 333
    self.sr = 48000
    self.channel = 1
    self.shift_pct = 0.4
    self.apply_augmentations = apply_augmentations

    self.indices = indices if indices is not None else list(range(len(self.audio_files)))

    file_labels = sorted({os.path.basename(f).split('_')[0] for f in self.audio_files})
    if class_to_idx is not None:
      missing = sorted(set(file_labels) - set(class_to_idx.keys()))
      if missing:
        raise ValueError(f"Provided class_to_idx is missing labels: {missing}")
      self.class_to_idx = dict(class_to_idx)
    else:
      self.class_to_idx = {name: i for i, name in enumerate(file_labels)}
        
  def __len__(self):
    return len(self.indices)    
    
  def __getitem__(self, idx):
    file_index = self.indices[idx]
    audio_file = self.audio_files[file_index]

    base_name = os.path.basename(audio_file)
    class_name = base_name.split('_')[0]
    class_id = self.class_to_idx[class_name]

    y, sr = librosa.load(audio_file, sr=48000, mono=True)
    # Ensure a consistent duration for every sample so that DataLoader can batch them.
    # `self.duration` represents milliseconds.
    target_num_samples = int(self.sr * (self.duration / 1000.0))
    if len(y) < target_num_samples:
      pad_amount = target_num_samples - len(y)
      y = np.pad(y, (0, pad_amount))
    elif len(y) > target_num_samples:
      y = y[:target_num_samples]

    # Apply time shift augmentation on the waveform (if enabled)
    if self.apply_augmentations:
      y = audio_util.time_shift(y, 20)

    # Linear spectrogram
    spectrogram = audio_util.create_linear_spectrogram(y, sr)

    # Apply spectrogram augmentations (if enabled)
    if self.apply_augmentations:
      spectrogram = audio_util.time_mask(spectrogram)
      spectrogram = audio_util.freq_mask(spectrogram)

    spec_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)
    return spec_tensor, class_id
  
def main():
  dataset = AudioDatasetLoader("./output")

  for i in range(0, len(dataset), 20):
      spectrogram, class_id = dataset[i]

      # fig, ax = plt.subplots()
      # img = librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=48000, fmax=8000, ax=ax)
      # fig.colorbar(img, ax=ax, format='%+2.0f dB')
      # ax.set(title=f"Mel-frequency spectrogram - {class_id}")
      # plt.show()

  # Randomly generate 80:20 split between training and validation
  num_items = len(dataset)
  num_train = round(num_items * 0.8)
  num_val = num_items - num_train
  train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

  # Create training and validation data loaders
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

  
  for X, y in test_dataloader:
      print(f"Shape of X [N, C, H, W]: {X.shape}")
      print(f"Shape of y: {y.shape} {y.dtype}")
      break

if __name__ == "__main__":
  main()