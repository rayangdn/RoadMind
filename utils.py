import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_examples(model, data_loader, device, num_samples=4, save_dir='./logs', testing=False,
                  use_depth_aux=False, use_semantic_aux=False):
    
    # Get a batch of data
    data_batch = next(iter(data_loader))
    
    # Move data to device
    camera = data_batch['camera'].to(device)
    history = data_batch['history'].to(device)
    if not testing:
        future = data_batch['future'].to(device)
    
    # Get depth and semantic data if the flags are True
    depth_gt = data_batch['depth'].to(device) if use_depth_aux else None
    semantic_gt = data_batch['semantic_label'].to(device) if use_semantic_aux else None
    
    # Define inverse normalization transform (matching visualize_samples)
    inverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Define a colormap for semantic visualization
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
    
    # Create semantic color mapping function
    def apply_semantic_colormap(semantic_tensor):
        # Convert tensor to numpy array
        semantic_np = semantic_tensor.cpu().numpy()
        height, width = semantic_np.shape
        
        # Create RGB image
        colored_semantic = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colormap
        for class_id, color in semantic_colormap.items():
            colored_semantic[semantic_np == class_id] = color
            
        return colored_semantic
    
    # Run model inference
    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(camera, history)
        
        # Check if model returns multiple outputs (main task + auxiliary tasks)
        if isinstance(outputs, list):
            traj_pred = outputs[0]
            depth_pred = outputs[1] if len(outputs) > 1 and use_depth_aux else None
            semantic_pred = outputs[2] if len(outputs) > 2 and use_semantic_aux else None
        else:
            traj_pred = outputs
            depth_pred = None
            semantic_pred = None
    
    # Convert tensors to numpy for plotting
    camera_np = camera.cpu()
    history_np = history.cpu().numpy()
    traj_pred_np = traj_pred.cpu().numpy()
    
    if not testing:
        future_np = future.cpu().numpy()
    
    # Randomly select samples to display
    selected_indices = random.sample(range(len(camera)), num_samples)
    
    # Determine number of rows needed based on the auxiliary flags
    num_rows = 2  # Always show camera and trajectory
    if use_depth_aux:
        num_rows += 2  # Add rows for GT depth and predicted depth
    if use_semantic_aux:
        num_rows += 2  # Add rows for GT semantic and predicted semantic
        
    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(4*num_samples, 3*num_rows))
    
    # Process each selected sample
    for i, idx in enumerate(selected_indices):
        row = 0
        
        # Plot camera view (top row)
        camera_img = inverse_normalize(camera_np[idx]).clamp(0, 1).permute(1, 2, 0)
        axes[row, i].imshow(camera_img)
        axes[row, i].set_title(f"Camera View {i+1}")
        axes[row, i].axis("off")
        row += 1
        
        # Plot trajectory (second row)
        axes[row, i].plot(history_np[idx, :, 0], history_np[idx, :, 1], "o-", color="gold", 
                        linewidth=2, label="Past")
        if not testing:
            axes[row, i].plot(future_np[idx, :, 0], future_np[idx, :, 1], "o-", color="green", 
                            linewidth=2, label="Ground Truth")
        axes[row, i].plot(traj_pred_np[idx, :, 0], traj_pred_np[idx, :, 1], "o-", color="red", 
                        linewidth=2, label="Predicted")
        axes[row, i].set_title(f"Trajectory {i+1}")
        axes[row, i].legend(loc='upper right')
        axes[row, i].axis("equal")
        axes[row, i].grid(True, linestyle='--', alpha=0.7)
        row += 1
        
        # Plot depth if the flag is True
        if use_depth_aux and depth_gt is not None and depth_pred is not None:
            # Ground truth depth
            depth_gt_img = depth_gt[idx].cpu().squeeze().numpy()
            axes[row, i].imshow(depth_gt_img)
            axes[row, i].set_title(f"GT Depth {i+1}")
            axes[row, i].axis("off")
            row += 1
            
            # Predicted depth
            depth_pred_img = depth_pred[idx].cpu().squeeze().numpy()
            axes[row, i].imshow(depth_pred_img)
            axes[row, i].set_title(f"Pred Depth {i+1}")
            axes[row, i].axis("off")
            row += 1
            
        # Plot semantic if the flag is True
        if use_semantic_aux and semantic_gt is not None and semantic_pred is not None:
            # Ground truth semantic
            semantic_gt_img = apply_semantic_colormap(semantic_gt[idx])
            axes[row, i].imshow(semantic_gt_img)
            axes[row, i].set_title(f"GT Semantic {i+1}")
            axes[row, i].axis("off")
            row += 1
            
            # Predicted semantic (get most likely class for each pixel)
            _, semantic_pred_classes = torch.max(semantic_pred[idx], dim=0)
            semantic_pred_img = apply_semantic_colormap(semantic_pred_classes)
            axes[row, i].imshow(semantic_pred_img)
            axes[row, i].set_title(f"Pred Semantic {i+1}")
            axes[row, i].axis("off")
            row += 1
    
    # Adjust layout and save
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"{save_dir}/prediction_examples.png")
    plt.close()
    
    