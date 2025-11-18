# Acoustic Side-Channel Attack on Keyboards

## Introduction

Acoustic side-channel attacks (ASCAs) targeting keyboards represent a persistent and practical threat to information security. While recent state-of-the-art research has demonstrated high classification accuracy using complex deep learning architectures like Transformers, the foundational impact of 2D feature representation choices remains under-investigated.

This project investigates the role of feature representation for a baseline Convolutional Neural Network (CNN) by comparing the performance of Standard (Linear) Spectrograms against perceptually-scaled Mel-Spectrograms, and evaluating the impact of SpecAugment data augmentation. Our results reveal that while Mel-spectrograms provide a substantial 10% increase in accuracy over linear representation in initial training, the highest overall test accuracy of 91% was achieved by a Standard Spectrogram model regularized with SpecAugment and trained for an extended duration (500 epochs). The findings suggest that a carefully tuned feature pipeline can enable a relatively simple, fast-training CNN to achieve performance comparable to more complex architectures—showcasing the practical viability of keyboard ASCAs.

## Project Structure

```
.
├── classification_model.py    # CNN model definition and training pipeline
├── dataloader.py              # Dataset loader with spectrogram generation
├── audio_util.py              # Audio processing utilities (spectrograms, augmentations)
├── dataset_generate.py        # Script for processing raw audio files into samples
├── dataset/                   # Raw audio dataset directory
├── output/                    # Processed audio samples
├── checkpoints/               # Saved model checkpoints
└── visualisations/            # Training plots and confusion matrices
```

## Installation

### Prerequisites

- Python 3.8+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd elec5305-project-520452275
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install librosa soundfile numpy matplotlib
pip install scikit-learn seaborn
```

## Usage

### 1. Prepare Dataset

Place your raw audio files in the `dataset/` directory. The audio files should be WAV format recordings of keyboard keystrokes.

### 2. Generate Processed Samples

Process raw audio files into individual keystroke samples:

```bash
python dataset_generate.py
```

This script uses onset detection to segment continuous audio into individual keystroke samples and saves them to the `output/` directory.

### 3. Train the Model

Train the CNN classifier:

```bash
python classification_model.py
```

The training script will:
- Load audio samples from `output/`
- Split data into train/validation/test sets (70:15:15)
- Apply SpecAugment augmentations during training
- Train the model and save checkpoints to `checkpoints/`
- Generate a confusion matrix visualization

### Configuration

Key parameters can be modified in `classification_model.py`:
- `num_epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size for training (default: 16)
- `data_path`: Path to processed audio samples (default: "./output")

### Feature Representation

The current implementation uses Linear Spectrograms by default. To switch to Mel-spectrograms, modify `dataloader.py`:

```python
# In AudioDatasetLoader.__getitem__:
# Replace:
spectrogram = audio_util.create_linear_spectrogram(y, sr)

# With:
spectrogram = audio_util.create_mel(y, sr)
```

### Data Augmentation: Time Shift and SpecAugment

The project implements two types of data augmentation:

1. **Time Shift**: Applied to the raw waveform before spectrogram generation
2. **SpecAugment**: Applied to the spectrogram (time masking and frequency masking)

#### Enabling/Disabling Augmentations

Augmentations are controlled by the `apply_augmentations` parameter when creating the dataset. By default, augmentations are enabled for training and disabled for validation/test:

```python
# In classification_model.py:
train_dataset = dataloader.AudioDatasetLoader(
    data_path, indices=train_indices, 
    apply_augmentations=True,  # Enable augmentations for training
    class_to_idx=class_to_idx
)
val_dataset = dataloader.AudioDatasetLoader(
    data_path, indices=val_indices, 
    apply_augmentations=False,  # Disable for validation
    class_to_idx=class_to_idx
)
```

#### Configuring Time Shift

Time shift is applied in `dataloader.py` (line 55). To modify the shift amount, change the second argument:

```python
# In AudioDatasetLoader.__getitem__:
if self.apply_augmentations:
    y = audio_util.time_shift(y, 20)  # Default: 20 (20% of signal length)
```

The second parameter (`shift_limit`) determines the maximum percentage of the signal that can be shifted (0-100). A value of 20 means up to 20% of the signal length can be shifted. For stronger augmentation, increase this value (e.g., `30` or `40`).

#### Configuring SpecAugment

SpecAugment consists of time masking and frequency masking, applied in `dataloader.py` (lines 62-63). To customize the masking parameters, modify the function calls:

```python
# In AudioDatasetLoader.__getitem__:
if self.apply_augmentations:
    # Time masking: masks up to max_mask_pct% of time steps
    spectrogram = audio_util.time_mask(
        spectrogram, 
        max_mask_pct=0.2,    # Default: 0.2 (20% of time steps)
        n_time_masks=1       # Default: 1 mask per sample
    )
    
    # Frequency masking: masks up to max_mask_pct% of frequency bins
    spectrogram = audio_util.freq_mask(
        spectrogram, 
        max_mask_pct=0.2,    # Default: 0.2 (20% of frequency bins)
        n_freq_masks=1       # Default: 1 mask per sample
    )
```

**Parameters:**
- `max_mask_pct`: Maximum percentage of dimensions to mask (0.0-1.0)
- `n_time_masks`: Number of time masks to apply
- `n_freq_masks`: Number of frequency masks to apply