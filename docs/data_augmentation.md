---
title: Data Augmentation
layout: default
filename: feature_extration.md
--- 

REF: `dataloader.py`

Utilising the individual keystroke samples obtained in the [Segmentation Methodology](segmentation.md) we create a PyTorch dataloader and accompanying helper functions to augment and load our samples into our model. The dataset is split 80-20 to reserve 20% of the dataset for validation at a later stage. Each audio sample is converted into a Mel-Spectrogram and then augmented using Google Brain's SpecAugment strategy which modifies the resultant spectrogram by warping it in the time direction and masking random blocks of consecutive time steps and mel-frequency channels. This is done both to increase the size of our dataset - as each sample has several resultant augmented versions - but to also increase the robustness of our model and decrease the likelihood of overfitting by forcing the network to learn relevant features.



