from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.nn import init
import dataloader
try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn and/or seaborn not available. Confusion matrix will use basic matplotlib.")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class AudioClassifier (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.bn1, self.relu1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.bn2, self.relu2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.bn3, self.relu3]

        # Forth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.bn4, self.relu4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

# ----------------------------
# Training Loop
# ----------------------------
def normalize_batch(inputs, mean, std):
    mean = mean[None, :, None, None].to(inputs.device)
    std = std[None, :, None, None].to(inputs.device)
    return (inputs - mean) / std.clamp_min(1e-6)


def compute_normalization_stats(dataloader):
    channel_sum = None
    channel_sum_sq = None
    total_pixels = 0

    for inputs, _ in dataloader:
        inputs = inputs.float()
        if channel_sum is None:
            channels = inputs.size(1)
            channel_sum = torch.zeros(channels, device=inputs.device)
            channel_sum_sq = torch.zeros(channels, device=inputs.device)

        channel_sum += inputs.sum(dim=(0, 2, 3))
        channel_sum_sq += (inputs ** 2).sum(dim=(0, 2, 3))
        total_pixels += inputs.numel() // inputs.size(1)

    mean = channel_sum / total_pixels
    std = (channel_sum_sq / total_pixels - mean ** 2).clamp(min=1e-6).sqrt()
    return mean.cpu(), std.cpu()


def evaluate(model, dataloader, mean, std):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = normalize_batch(inputs.to(device), mean, std)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def training(model, train_dl, val_dl, num_epochs, mean, std, checkpoint_path=None, class_to_idx=None, history_path=None):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  best_val_acc = float("-inf")
  checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
  history_path = Path(history_path) if history_path else None
  history = []

  # Repeat for each epoch
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):  
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = normalize_batch(inputs, mean, std)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    val_loss, val_acc = evaluate(model, val_dl, mean, std)
    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.2f}, Train Acc: {acc:.2f}, Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.2f}')
    history.append(
        {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "train_acc": float(acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
    )

    if checkpoint_path and val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "mean": mean.cpu(),
                "std": std.cpu(),
                "class_to_idx": class_to_idx,
                "history": history,
            },
            checkpoint_path,
        )
        print(f"Saved new best model checkpoint to {checkpoint_path}")

  if history_path:
      history_path.parent.mkdir(parents=True, exist_ok=True)
      with history_path.open("w") as f:
          json.dump(history, f, indent=2)
      print(f"Saved training history to {history_path}")

  print('Finished Training')
  return best_val_acc, history
  

def plot_confusion_matrix(model, dataloader, mean, std, idx_to_class, output_path=None):
    """Compute and plot confusion matrix for the model on the given dataloader."""
    if dataloader is None:
        print("No dataloader provided; skipping confusion matrix.")
        return
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = normalize_batch(inputs.to(device), mean, std)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Get class names in order
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Compute confusion matrix
    if HAS_SKLEARN:
        cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
    else:
        # Manual computation if sklearn not available
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true_label, pred_label in zip(all_labels, all_predictions):
            cm[true_label, pred_label] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    
    if HAS_SKLEARN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Count')
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    return cm


def main():
    data_path = "./output"
    base_dataset = dataloader.AudioDatasetLoader(data_path, apply_augmentations=False)
    class_to_idx = base_dataset.class_to_idx
    # for i in range(0, len(base_dataset), 20):
    #     spectrogram, class_id = base_dataset[i]
    #     fig, ax = plt.subplots()
    #     img = librosa.display.specshow(spectrogram.squeeze(0), x_axis='time', y_axis='mel', sr=48000, fmax=8000, ax=ax)
    #     fig.colorbar(img, ax=ax, format='%+2.0f dB')
    #     ax.set(title=f"Mel-frequency spectrogram - {class_id}")
    #     plt.show()

    # Randomly generate 70:15:15 split between training, validation and test
    num_items = len(base_dataset)
    if num_items < 3:
        raise ValueError("Dataset must contain at least 3 samples to perform train/val/test split.")

    num_train = max(1, int(num_items * 0.7))
    num_val = max(1, int(num_items * 0.15))
    num_test = num_items - num_train - num_val

    if num_test <= 0:
        num_test = 1
        if num_val > 1:
            num_val -= 1
        elif num_train > 1:
            num_train -= 1
        else:
            raise ValueError("Unable to create non-empty train/val/test splits with the current dataset size.")

    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(num_items, generator=generator).tolist()

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:num_train + num_val]
    test_indices = all_indices[num_train + num_val:]

    train_dataset = dataloader.AudioDatasetLoader(
        data_path, indices=train_indices, apply_augmentations=True, class_to_idx=class_to_idx
    )
    val_dataset = dataloader.AudioDatasetLoader(
        data_path, indices=val_indices, apply_augmentations=False, class_to_idx=class_to_idx
    )
    test_dataset = dataloader.AudioDatasetLoader(
        data_path, indices=test_indices, apply_augmentations=False, class_to_idx=class_to_idx
    )
    stats_dataset = dataloader.AudioDatasetLoader(
        data_path, indices=train_indices, apply_augmentations=False, class_to_idx=class_to_idx
    )

    # Create training, validation and test data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    stats_loader = DataLoader(stats_dataset, batch_size=32, shuffle=False)

    train_mean, train_std = compute_normalization_stats(stats_loader)

    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break


    # Create the model and put it on the GPU if available, otherwise CPU
    model = AudioClassifier(num_classes=num_classes).to(device)
    num_epochs=200   # Adjust as needed.

    checkpoint_path = Path("checkpoints") / "audio_classifier.pt"
    history_path = Path("checkpoints") / "training_history.json"
    best_val_acc, _ = training(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs,
        train_mean,
        train_std,
        checkpoint_path=checkpoint_path,
        class_to_idx=class_to_idx,
        history_path=history_path,
    )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_mean = checkpoint.get("mean", train_mean)
        best_std = checkpoint.get("std", train_std)
        saved_mapping = checkpoint.get("class_to_idx")
        if saved_mapping is not None:
            class_to_idx = saved_mapping
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        print(f"Loaded best model from checkpoint with Val Acc: {checkpoint.get('val_acc', best_val_acc):.2f}")
    else:
        best_mean, best_std = train_mean, train_std

    test_loss, test_acc = evaluate(model, test_dataloader, best_mean, best_std)
    print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}")

    # Plot confusion matrix
    confusion_matrix_path = Path("visualisations") / "confusion_matrix.png"
    plot_confusion_matrix(model, test_dataloader, best_mean, best_std, idx_to_class, 
                         output_path=confusion_matrix_path)
   
if __name__ == "__main__":
   main()


