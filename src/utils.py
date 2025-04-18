import torch
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth"):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from a checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: Resuming from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, None  # Start from epoch 0 if no checkpoint is found
    
def plot_results(train_losses, val_losses, val_ade, val_fde):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Losses
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].legend()
    axs[0].grid(True)

    # Plot ADE and FDE
    axs[1].plot(val_ade, label='ADE')
    axs[1].plot(val_fde, label='FDE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Displacement Error')
    axs[1].set_title('Validation ADE and FDE')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('../outputs/training_metrics.png')
    plt.close()
