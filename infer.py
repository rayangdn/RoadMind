import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from data.data_loader import NuplanDataLoader, AugmentedNuPlanDataset, get_data_paths
from model import LightningRoadMind

def evaluate(model, test_loader, submission_dir, device):
    model.eval()
    all_plans = []
    
    with torch.no_grad():
        for batch in test_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            
            # Forward pass
            outputs = model(camera, history)
            
            # Extract trajectory prediction from outputs
            if isinstance(outputs, list):
                # Multi-task model returns a list with trajectory as first element
                pred_future = outputs[0]
            else:
                # Single-task model returns only trajectory
                pred_future = outputs
            
            # Add to collection (only position coordinates, not heading)
            all_plans.append(pred_future.cpu().numpy()[..., :2])
            
    all_plans = np.concatenate(all_plans, axis=0)
    
    # Now save the plans as a csv file
    pred_xy = all_plans  # shape: (total_samples, T, 2)

    # Flatten to (total_samples, T*2)
    total_samples, T, D = pred_xy.shape
    pred_xy_flat = pred_xy.reshape(total_samples, T * D)

    # Build a DataFrame with an ID column
    ids = np.arange(total_samples)
    df_xy = pd.DataFrame(pred_xy_flat)
    df_xy.insert(0, "id", ids)

    # Column names: id, x_1, y_1, x_2, y_2, ..., x_T, y_T
    new_col_names = ["id"]
    for t in range(1, T + 1):
        new_col_names.append(f"x_{t}")
        new_col_names.append(f"y_{t}")
    df_xy.columns = new_col_names

    # Save to CSV
    df_xy.to_csv(os.path.join(submission_dir, "submission_test.csv"), index=False)

    print(f"Shape of df_xy: {df_xy.shape}")
    
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data directories
    data_dir = './data'
    submission_dir = './submission'
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    # Download and extract dataset
    #data_loader = NuplanDataLoader(data_dir=data_dir)
    data_paths = get_data_paths(data_dir)
    
    # Create dataset
    test_dataset = AugmentedNuPlanDataset(
        data_paths['test'], 
        test=True,
        include_dynamics=True,
        augment_prob=0.0
    )
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model - note that the checkpoint might contain use_depth_aux and use_semantic_aux params
    checkpoint_path = "checkpoints/roadmind/roadmind_22_epoch=98_val_ade=1.72.ckpt"
    model = LightningRoadMind.load_from_checkpoint(checkpoint_path).to(device)
    
    # Evaluate model
    evaluate(model, test_loader, submission_dir, device)
    
if __name__ == "__main__":
    main()