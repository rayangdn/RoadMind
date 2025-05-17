import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_examples(model, data_loader, device, num_samples=4, save_dir='./logs', testing=False,):
    
    # Get a batch of data
    data_batch = next(iter(data_loader))
    
    # Move data to device
    camera = data_batch['camera'].to(device)
    history = data_batch['history'].to(device)
    if not testing:
        future = data_batch['future'].to(device)
    
    # Define inverse normalization transform (matching visualize_samples)
    inverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Run model inference
    model.eval()
    model.to(device)
    with torch.no_grad():
        traj_pred = model(camera, history)
        
    # Convert tensors to numpy for plotting
    camera_np = camera.cpu()
    history_np = history.cpu().numpy()
    traj_pred_np = traj_pred.cpu().numpy()
    
    if not testing:
        future_np = future.cpu().numpy()
    
    # Randomly select samples to display
    selected_indices = random.sample(range(len(camera)), num_samples)
     
    # Create figure and axes
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 6))
    
    # Process each selected sample
    for i, idx in enumerate(selected_indices):
        
        # Plot camera view (top row)
        camera_img = inverse_normalize(camera_np[idx]).clamp(0, 1).permute(1, 2, 0)
        axes[0, i].imshow(camera_img)
        axes[0, i].set_title(f"Camera View {i+1}")
        axes[0, i].axis("off")
        
        # Plot trajectory (second row)
        axes[1, i].plot(history_np[idx, :, 0], history_np[idx, :, 1], "o-", color="gold", 
                        linewidth=2, label="Past")
        if not testing:
            axes[1, i].plot(future_np[idx, :, 0], future_np[idx, :, 1], "o-", color="green", 
                            linewidth=2, label="Ground Truth")
        axes[1, i].plot(traj_pred_np[idx, :, 0], traj_pred_np[idx, :, 1], "o-", color="red", 
                        linewidth=2, label="Predicted")
        axes[1, i].set_title(f"Trajectory {i+1}")
        axes[1, i].legend(loc='upper right')
        axes[1, i].axis("equal")
        axes[1, i].grid(True, linestyle='--', alpha=0.7)
        
    # Adjust layout and save
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"{save_dir}/prediction_examples.png")
    plt.close()
    
    