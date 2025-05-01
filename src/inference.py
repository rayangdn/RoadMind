import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import NuplanDataLoader, NuPlanDataset
from model import RoadMind
from utils import plot_examples, visualize_attention

def main():
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data directories
    data_dir = '../data'
    model_dir = '../model'
    output_dir = '../outputs'
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'examples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Download and extract dataset
    data_loader = NuplanDataLoader(data_dir=data_dir, download=False)
    data_paths = data_loader.get_data_paths()
    
    # Create datasets
    val_dataset = NuPlanDataset(data_paths['val'])
    
    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    model = RoadMind()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    
    plot_examples(model, val_loader, device, os.path.join(output_dir, 'examples'), num_samples=3)
    
    # Generate visualizations
    visualize_attention(model, val_loader, device,  os.path.join(output_dir, 'logs'),num_samples=3, random_samples=True)
    
    
if __name__ == "__main__":
    main()