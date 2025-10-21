---
title: Segmentation Methodology
layout: default
filename: segmentation.md
--- 

REF: `dataset_generate.py`

The provided dataset from the reference paper contains a set of 36 audio samples (representing keys 0-9, a-z) with each sample containing 25 individual presses taken with ample time in between samples to allow for resonants to subside. The following section utilises Python audio analysis library - "Librosa" - to characterise and segment each sample into isolated, keystrokes to later be processed into a corresponding Mel-Spectrogram.

The original dataset was imported into Librosa as mono-channel 48kHz samples to account for the broadband nature of keystrokes which tend to include a high-frequency component at ~18kHz but varies depending on the switch type. 

Peak's are determined by first generating a spectral flux onset strength envelope utilising librosa's `onset_strength()` function. The value for the onset strength at any given time $t$ is determined with the following equation:
$$mean_{f} \ max(0,S[f, t] - ref[f,t-lag])$$
where the value $ref$ is $S$ after local max filtering along the frequency axis.

Peak's are then determined using the `onset_detect()` function with the following parameters:
- `min_time=10`

The back-tracked resulting return value is converted from frames to samples, and a 1440 (0.33s) sample is taken from each trough to form an individual segment. This process is repeated across the entire dataset to generate `\output` - which contains 25 individual presses for each key for further analysis.