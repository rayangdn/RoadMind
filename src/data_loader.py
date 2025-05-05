import os
import gdown
import zipfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NuplanDataLoader:
    def __init__(self, data_dir='../data', download=False):
        self.data_dir = data_dir
        self.file_urls = {
            'train': "https://drive.google.com/uc?id=1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr",
            'val': "https://drive.google.com/uc?id=1wtmT_vH9mMUNOwrNOMFP6WFw6e8rbOdu",
            'test': "https://drive.google.com/uc?id=1G9xGE7s-Ikvvc2-LZTUyuzhWAlNdLTLV"
        }
        
        # Download data if requested and not already present
        if download:
            self.download_dataset()
    
    def download_dataset(self):
        print("Downloading and extracting nuPlan dataset...")
        
        for split, url in self.file_urls.items():
            output_zip = os.path.join(self.data_dir, f"dlav_{split}.zip")
            extracted_dir = os.path.join(self.data_dir, f"dlav_{split}")
            
            # Check if already extracted
            if os.path.exists(extracted_dir):
                print(f"{split} data already exists at {extracted_dir}")
                continue
            
            # Download if zip doesn't exist
            if not os.path.exists(output_zip):
                print(f"Downloading {split} data...")
                gdown.download(url, output_zip, quiet=False)

            # Extract
            print(f"Extracting {split} data...")
            with zipfile.ZipFile(output_zip , 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            print(f"{split} data downloaded and extracted successfully")

    def get_data_paths(self):
        return {
            'train': os.path.join(self.data_dir, "train"),
            'val': os.path.join(self.data_dir, "val"),
            'test': os.path.join(self.data_dir, "test_public")
        }
    
class AugmentedNuPlanDataset(Dataset):
    def __init__(self, data_path, test=False, validation=False, augment_prob=0.5):
        self.data_path = data_path
        self.test = test
        self.augment_prob = augment_prob if not (test or validation) else 0
        self.validation = validation
        self.command_mapping = {'forward': 0, 'left': 1, 'right': 2}
        if test:
            self.samples = [os.path.join(data_path, fn) for fn in sorted([f for f in os.listdir(data_path) if f.endswith(".pkl")], 
                                                                          key=lambda fn: int(os.path.splitext(fn)[0]))]
        else:
            self.samples = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        
        # Define normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
        
        # Define transform
        self.augmentations = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.7),  
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.7),  
        ])
        
        self.resize = transforms.Resize((224, 224), antialias=True)
 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load pickle file
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)
        
        
        # Extract camera, history and future features
        camera = torch.FloatTensor(data['camera']).permute(2, 0, 1) / 255.0
        history = torch.FloatTensor(data['sdc_history_feature'])
        if not self.test:
            future = torch.FloatTensor(data['sdc_future_feature'])

        # Resize camera
        camera = self.resize(camera)
        
        # Normalize camera
        camera = self.normalize(camera)  
        
        # Apply random flip to camera and history features
        if random.random() <= self.augment_prob:
            camera = self.augmentations(camera)
            camera = torch.flip(camera, dims=[2]) # Flip horizontally
            history[:, 1] = -history[:, 1] # Flip y-coordinates
            if not self.test:
                future[:, 1] = -future[:, 1]
                
        # Create sample dictionary
        if self.test:
            sample = {
                'camera': camera,
                'sdc_history_feature': history
            }
        else:

            sample = {
                'camera': camera,
                'sdc_history_feature': history,
                'sdc_future_feature': future,
            }
        return sample


def visualize_samples(dataset, num_samples=4):
    
    # Define the inverse normalization transform
    inverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    inverse_resize = transforms.Resize((200, 300), antialias=True)
    
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    # Process each sample
    for i, idx in enumerate(sample_indices):
        data = dataset[idx]
        
        # Plot camera view (top row)
        camera_img = inverse_resize(data["camera"])
        camera_img = inverse_normalize(camera_img).clamp(0, 1).permute(1, 2, 0) 
        axes[0, i].imshow(camera_img)
        axes[0, i].set_title(f"Camera View {i+1}")
        axes[0, i].axis("off")
        
        # Plot trajectory (bottom row)
        history = data["sdc_history_feature"]
        future = data["sdc_future_feature"]
        
        axes[1, i].plot(history[:, 0], history[:, 1], "o-", color="gold", linewidth=2, label="Past")
        axes[1, i].plot(future[:, 0], future[:, 1], "o-", color="green", linewidth=2, label="Future")
        axes[1, i].set_title(f"Trajectory {i+1}")
        axes[1, i].legend(loc='upper right')
        axes[1, i].axis("equal")
        axes[1, i].grid(True, linestyle='--', alpha=0.7)
        
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
# nuplan_loader = NuplanDataLoader(data_dir='../data', download=False)
    
# # Get data paths
# data_paths = nuplan_loader.get_data_paths()

# # Visualize samples from the training set
# dataset = AugmentedNuPlanDataset(data_paths['train'], test=False, validation=False, augment_prob=0.5)
# visualize_samples(dataset, num_samples=4)

    
# # Function to calculate mean and std for the entire dataset
# def calculate_mean_std(dataset, batch_size=32, num_workers=4):
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )
    
#     sum_tensor = torch.zeros(3)
#     sum_squares_tensor = torch.zeros(3)
#     pixel_count = 0
    
#     # Process ALL images in the ENTIRE dataset
#     for batch in tqdm.tqdm(dataloader, desc="Calculating mean and std"):
#         images = batch['camera']  # Shape: [batch_size, 3, H, W]
#         batch_size, channels, height, width = images.shape
        
#         # Sum across all dimensions except the channel dimension
#         sum_tensor += torch.sum(images, dim=[0, 2, 3])
#         sum_squares_tensor += torch.sum(images ** 2, dim=[0, 2, 3])
        
#         # Count the total number of pixels
#         pixel_count += batch_size * height * width
    
#     # Calculate mean and std for the ENTIRE dataset
#     mean = sum_tensor / pixel_count
#     var = (sum_squares_tensor / pixel_count) - (mean ** 2)
#     std = torch.sqrt(var)
    
#     return mean, std

# # Calculate mean and std for the entire dataset
# mean, std = calculate_mean_std(temp_dataset)
# print(f"Dataset mean: {mean}")
# print(f"Dataset std: {std}")