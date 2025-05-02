import os
import gdown
import zipfile
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class NuplanDataLoader:
    """
    Class to handle downloading, extracting, and loading the nuPlan dataset.
    """
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
    
class NuPlanDataset(Dataset):
    def __init__(self, data_path, testing=False):
        self.data_path = data_path
        self.testing = testing
        self.command_mapping = {'forward': 0, 'left': 1, 'right': 2}
        if testing:
            self.samples = [os.path.join(data_path, fn) for fn in sorted([f for f in os.listdir(data_path) if f.endswith(".pkl")], 
                                                                          key=lambda fn: int(os.path.splitext(fn)[0]))]
        else:
            self.samples = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]

        self.camera_transform = CameraTransform(normalize=True) 
           
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load pickle file
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)
        
        # Extract and transform camera data
        camera = data['camera'].astype(np.float32)
        camera = np.transpose(camera, (2, 0, 1))  # CHW format
        camera = self.camera_transform(camera)
        
        # Extract motion history
        sdc_history_feature = data['sdc_history_feature'].astype(np.float32)
        
        # Create sample dictionary
        if self.testing:
            sample = {
                'camera': camera,
                'sdc_history_feature': torch.from_numpy(sdc_history_feature)
            }
        else:
            sdc_future_feature = data['sdc_future_feature'].astype(np.float32)
            
            sample = {
                'camera': camera,
                'sdc_history_feature': torch.from_numpy(sdc_history_feature),
                'sdc_future_feature': torch.from_numpy(sdc_future_feature),
            }

        return sample
    
class CameraTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean = torch.tensor([0.587, 0.605, 0.590])
        self.std = torch.tensor([0.132, 0.125, 0.163])
        
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Basic normalization (0-1)
        if image.max() > 1.0:
            image = image / 255.0
        
        if self.normalize:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            image = (image - mean) / std
        return image

# from torch.utils.data import DataLoader
# import tqdm

# data_loader = NuplanDataLoader(data_dir='../data', download=False)
# data_paths = data_loader.get_data_paths()
# temp_dataset = NuPlanDataset(data_paths['train'])
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