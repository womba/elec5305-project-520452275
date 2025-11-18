import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(json_path: Path):
    with json_path.open("r") as f:
        history = json.load(f)
    return history


def plot_training_history(json_path: str = "linear_spec_4096.json", output_dir: str = "visualisations"):
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(json_path)

    # Expect list of dicts with keys: epoch, train_loss, train_acc, val_loss, val_acc
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss") for h in history]
    val_loss = [h.get("val_loss") for h in history]
    train_acc = [h.get("train_acc") for h in history]
    val_acc = [h.get("val_acc") for h in history]

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = output_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    print(f"Saved loss curve to: {loss_path}")
    print(f"Saved accuracy curve to: {acc_path}")


if __name__ == "__main__":
    plot_training_history()


