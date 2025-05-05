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
    
    
def plot_results(train_losses, val_losses, val_ade, val_fde, 
                val_depth_losses, val_semantic_losses,
                output_dir='../outputs/logs'):
    """
    Plot training and validation metrics, including auxiliary task losses
    """
    # Create a 4-panel figure
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    
    # Plot 1: Overall Losses
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Trajectory metrics (ADE and FDE)
    axs[1].plot(val_ade, label='ADE')
    axs[1].plot(val_fde, label='FDE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Displacement Error')
    axs[1].set_title('Validation ADE and FDE')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: Depth loss
    axs[2].plot(val_depth_losses, label='Depth Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('L1 Loss')
    axs[2].set_title('Validation Depth Estimation Loss')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot 4: Semantic loss
    axs[3].plot(val_semantic_losses, label='Semantic Loss')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Cross-Entropy Loss')
    axs[3].set_title('Validation Semantic Segmentation Loss')
    axs[3].legend()
    axs[3].grid(True)
    
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

def plot_examples(model, data_loader, device, output_dir='../outputs/examples', 
                 num_samples=4, testing=False):
    """
    Visualize model predictions, including depth and semantic segmentation
    """
    # Get batch of data
    data_batch_zero = next(iter(data_loader))
    camera = data_batch_zero['camera'].to(device)
    history = data_batch_zero['sdc_history_feature'].to(device)
    
    if not testing:
        future = data_batch_zero['sdc_future_feature'].to(device)
        future_np = future.cpu().numpy()
    
    # Get ground truth for auxiliary tasks
    depth_gt = data_batch_zero['depth'].to(device)
    semantic_gt = data_batch_zero['semantic_label'].to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        pred_future, pred_depth, pred_semantic = model(camera, history)
    
    # Convert tensors to numpy for plotting
    camera_np = unormalize_image(camera).cpu().numpy()
    history_np = history.cpu().numpy()
    pred_future_np = pred_future.cpu().numpy()
    
    # Convert auxiliary outputs
    depth_gt_np = depth_gt.cpu().numpy()
    pred_depth_np = pred_depth.cpu().numpy()
    
    semantic_gt_np = semantic_gt.cpu().numpy()
    pred_semantic_np = torch.argmax(pred_semantic, dim=1).cpu().numpy()
    
    # Select random samples to visualize
    k = num_samples
    selected_indices = random.choices(np.arange(len(camera_np)), k=k)
    
    # Define colormap for semantic segmentation
    semantic_colormap = {
        0: (0, 0, 0),         # UNLABELED
        1: (0, 0, 142),       # CAR
        2: (0, 0, 70),        # TRUCK
        3: (220, 20, 60),     # PEDESTRIAN
        4: (119, 11, 32),     # BIKE
        5: (152, 251, 152),   # TERRAIN
        6: (128, 64, 128),    # ROAD
        7: (244, 35, 232),    # SIDEWALK
        8: (70, 130, 180),    # SKY
        9: (250, 170, 30),    # TRAFFIC_LIGHT
        10: (190, 153, 153),  # FENCE
        11: (220, 220, 0),    # TRAFFIC_SIGN
        12: (255, 255, 255),  # LANE_LINE
        13: (55, 176, 189),   # CROSSWALK
        14: (0, 60, 100)      # BUS
    }
    
    # 1. Create figure for trajectories
    fig, axes = plt.subplots(2, k, figsize=(4*k, 8))
    
    for i, idx in enumerate(selected_indices):
        # Plot camera view in the top row
        axes[0, i].imshow(camera_np[idx].transpose(1, 2, 0))
        axes[0, i].set_title(f"Example {i+1}")
        axes[0, i].axis("off")
        
        # Plot trajectory in the bottom row
        axes[1, i].plot(history_np[idx, :, 0], history_np[idx, :, 1], "o-", color="gold", label="Past")
        if not testing:
            axes[1, i].plot(future_np[idx, :, 0], future_np[idx, :, 1], "o-", color="green", label="Future")
        axes[1, i].plot(pred_future_np[idx, :, 0], pred_future_np[idx, :, 1], "o-", color="red", label="Predicted")
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_examples.png")
    plt.close()
    
    # 2. Plot depth estimation
    fig, axes = plt.subplots(2, k, figsize=(4*k, 8))
    
    for i, idx in enumerate(selected_indices):
        # Ground truth depth
        axes[0, i].imshow(depth_gt_np[idx, :, :, 0], cmap='viridis')
        axes[0, i].set_title(f"GT Depth {i+1}")
        axes[0, i].axis("off")
        
        # Predicted depth
        axes[1, i].imshow(pred_depth_np[idx, :, :, 0], cmap='viridis')
        axes[1, i].set_title(f"Pred Depth {i+1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_examples.png")
    plt.close()
    
    # 3. Plot semantic segmentation
    fig, axes = plt.subplots(2, k, figsize=(4*k, 8))
    
    for i, idx in enumerate(selected_indices):
        # Ground truth semantic segmentation
        gt_semantic_rgb = np.zeros((semantic_gt_np[idx].shape[0], semantic_gt_np[idx].shape[1], 3), dtype=np.uint8)
        for class_id, color in semantic_colormap.items():
            mask = semantic_gt_np[idx] == class_id
            gt_semantic_rgb[mask] = color
        
        axes[0, i].imshow(gt_semantic_rgb)
        axes[0, i].set_title(f"GT Semantic {i+1}")
        axes[0, i].axis("off")
        
        # Predicted semantic segmentation
        pred_semantic_rgb = np.zeros((pred_semantic_np[idx].shape[0], pred_semantic_np[idx].shape[1], 3), dtype=np.uint8)
        for class_id, color in semantic_colormap.items():
            mask = pred_semantic_np[idx] == class_id
            pred_semantic_rgb[mask] = color
        
        axes[1, i].imshow(pred_semantic_rgb)
        axes[1, i].set_title(f"Pred Semantic {i+1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/semantic_examples.png")
    plt.close()
