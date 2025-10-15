from torch.utils.data import DataLoader, Dataset, random_split
import librosa
import os
import audio_util
import glob

class AudioDatasetLoader(Dataset):
  def __init__(self, data_path):
    self.data_path = data_path
    self.audio_files = sorted(glob.glob(os.path.join(data_path, "*.wav")))
    self.duration = 3333
    self.sr = 48000
    self.channel = 1
    self.shift_pct = 0.4
            
  def __len__(self):
    return len(self.audio_files)    
    
  def __getitem__(self, idx):
    audio_file = self.audio_files[idx]
    basename = os.path.basename(audio_file)
    class_id = basename.split('_')[0]

    y, sr = librosa.load(audio_file, sr=48000, mono=True)

    mel_spectrogram = audio_util.create_mel(y, sr)
    aug_spectrogram = audio_util.time_shift(mel_spectrogram, 20)
    aug_spectrogram = audio_util.time_mask(aug_spectrogram)
    aug_spectrogram = audio_util.freq_mask(aug_spectrogram)

    return aug_spectrogram, class_id
  
def main():
  dataset = AudioDatasetLoader("./output")

  for i in range(len(dataset)):
      spectrogram, class_id = dataset[i]
      print(f"Loaded {class_id} with shape {spectrogram.shape}")

if __name__ == "__main__":
  main()