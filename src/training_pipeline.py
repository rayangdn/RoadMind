import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from data_loader import NuplanDataLoader, NuPlanDataset
from model import RoadMind
from train import train_model 
from utils import plot_results, plot_examples

def main():
    
    # Set hyperparameters
    batch_size = 32
    num_epochs = 2
    learning_rate = 5e-4
    patience = 20
    
    # Auxiliary task settings
    use_depth_aux = True              # Whether to use depth estimation as auxiliary task
    use_semantic_aux = True          # Whether to use semantic segmentation as auxiliary task
    lambda_depth = 0.1                # Weight for depth loss
    lambda_semantic = 0.1             # Weight for semantic segmentation loss
    num_semantic_classes = 15         # Number of semantic classes
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data directories
    data_dir = '../data'
    output_dir = '../outputs'
    model_dir = '../model'
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'examples'), exist_ok=True)
    
    # Download and extract dataset
    data_loader = NuplanDataLoader(data_dir=data_dir, download=False) # Set download=True to download the dataset
    data_paths = data_loader.get_data_paths()
    
    # Create datasets
    train_dataset = NuPlanDataset(data_paths['train'])
    val_dataset = NuPlanDataset(data_paths['val'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model with auxiliary task support
    model = RoadMind(
        input_dim=3,
        hidden_dim=128,
        image_embed_dim=128,
        output_seq_len=60,
        num_layers=2,
        dropout_rate=0.3,
        bidirectional=True,
        use_depth_aux=use_depth_aux,
        use_semantic_aux=use_semantic_aux,
        num_semantic_classes=num_semantic_classes
    )
        
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)
    
    # Train model with auxiliary tasks
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        patience=patience,
        model_dir=model_dir,
        output_dir=os.path.join(output_dir, 'checkpoints'),
        device=device,
        use_depth_aux=use_depth_aux,
        use_semantic_aux=use_semantic_aux,
        lambda_depth=lambda_depth,
        lambda_semantic=lambda_semantic
    )
    
    # Unpack training results
    train_losses, val_losses, val_ade, val_fde, val_depth_losses, val_semantic_losses = train_results

    # Plot results
    plot_results(
    train_losses, val_losses, val_ade, val_fde, 
    val_depth_losses, val_semantic_losses,
    os.path.join(output_dir, 'logs')
    )
    
    print(f"Best ADE: {min(val_ade):.4f}")
    print(f"Best FDE: {min(val_fde):.4f}")
    
    if use_depth_aux:
        print(f"Best Depth Loss: {min(val_depth_losses):.4f}")
    
    if use_semantic_aux:
        print(f"Best Semantic Loss: {min(val_semantic_losses):.4f}")
        
    print(f"Training complete. Results saved to {os.path.join(output_dir, 'logs')}")
    
    # Plot examples with auxiliary task visualizations
    plot_examples(
    model, val_loader, device, 
    output_dir=os.path.join(output_dir, 'examples')
    )

if __name__ == "__main__":
    main()