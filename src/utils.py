import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth", store_checkpoint_for_every_epoch=False):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if store_checkpoint_for_every_epoch:
        # If the checkpoint has to be stored for every epoch to the name of the checkpoint at the end will be added _ep{epoch}
        # This is implemented since if you store all checkpoints to the same filename the next epoch will override the results of the previous epoch
        checkpoint_path = checkpoint_path[:checkpoint_path.rfind('.')] + f"_ep{epoch}" + checkpoint_path[checkpoint_path.rfind('.')+1:]
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
        print(f"Checkpoint loaded: Resuming from epoch {epoch+1} with loss {loss:.4f}")
        return epoch, loss
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, None  # Start from epoch 0 if no checkpoint is found
    
    
def plot_results(train_losses, val_losses, val_ade, val_fde, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def unormalize_image(normalized_image, mean=[0.587, 0.605, 0.589], std=[0.132, 0.125, 0.163]):

    # Convert lists to tensors if they aren't already
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=normalized_image.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=normalized_image.device)
    
    # Reshape mean and std for proper broadcasting
    if normalized_image.dim() == 3:  # (C, H, W)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif normalized_image.dim() == 4:  # (B, C, H, W)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    
    # Reverse the normalization: unnormalized = normalized * std + mean
    unnormalized = normalized_image * std + mean
    
    # Scale to [0, 255] range
    unnormalized = unnormalized.clamp(0, 1) * 255
    unnormalized = unnormalized.type(torch.uint8)  # Convert to uint8
    
    return unnormalized

def plot_examples(model, data_loader, device, output_dir='../outputs/examples', num_samples=4, testing=False):
    
    data_batch_zero = next(iter(data_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = data_batch_zero['camera'].to(device)
    history = data_batch_zero['sdc_history_feature'].to(device)
    if not testing:
        future = data_batch_zero['sdc_future_feature'].to(device)
        future = future.cpu().numpy()
    
    model.eval()
    with torch.no_grad():
        pred_future = model(camera, history)
    
    camera = unormalize_image(camera).cpu().numpy()  # corrected unormalization
    history = history.cpu().numpy()
    pred_future = pred_future.cpu().numpy()
    
    k = num_samples
    selected_indices = random.choices(np.arange(len(camera)), k=k)
    
    # Create one figure with 2 rows and k columns
    fig, axes = plt.subplots(2, k, figsize=(4*k, 8))
    
    for i, idx in enumerate(selected_indices):
        # Plot camera view in the top row
        axes[0, i].imshow(camera[idx].transpose(1, 2, 0))
        axes[0, i].set_title(f"Example {i+1}")
        axes[0, i].axis("off")
        
        # Plot trajectory in the bottom row
        axes[1, i].plot(history[idx, :, 0], history[idx, :, 1], "o-", color="gold", label="Past")
        if not testing:
            axes[1, i].plot(future[idx, :, 0], future[idx, :, 1], "o-", color="green", label="Future")
        axes[1, i].plot(pred_future[idx, :, 0], pred_future[idx, :, 1], "o-", color="red", label="Predicted")
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_examples.png")
    plt.close()