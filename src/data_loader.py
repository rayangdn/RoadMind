import os
import gdown
import zipfile
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset



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
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
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
    def __init__(self, data_path, testing=False, transform=None):
        self.data_path = data_path
        self.testing = testing
        self.transform = transform
        self.command_mapping = {'forward': 0, 'left': 1, 'right': 2}
        if testing:
            self.samples = [os.path.join(data_path, fn) for fn in sorted([f for f in os.listdir(data_path) if f.endswith(".pkl")], 
                                                                         key=lambda fn: int(os.path.splitext(fn)[0]))]
        else:
            self.samples = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        
        # Initialize transforms
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
    
        # Extract depth data
        # depth = data['depth'].astype(np.float32)
        # depth = depth.transpose(2, 0, 1)
        
        # Extract driving command data
        driving_command = self.command_mapping[data['driving_command']]
        
        #Extract semantic label
        # semantic_label = data['semantic_label'].astype(np.int64)
        
        # Create sample dictionary
        if self.testing:
            sample = {
                'camera': camera,
                # 'depth': torch.from_numpy(depth),
                'driving_command': torch.tensor(driving_command, dtype=torch.long),
                'sdc_history_feature': torch.from_numpy(sdc_history_feature)
            }
        else:
            sdc_future_feature = data['sdc_future_feature'].astype(np.float32)
            
            sample = {
                'camera': camera,
                # 'depth': torch.from_numpy(depth),
                'driving_command': torch.tensor(driving_command, dtype=torch.long),
                'sdc_history_feature': torch.from_numpy(sdc_history_feature),
                'sdc_future_feature': torch.from_numpy(sdc_future_feature),
                # 'semantic_label': torch.from_numpy(semantic_label)
            }

        return sample
    
class CameraTransform:
    """Process the camera data with improved normalization"""
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Basic normalization (0-1)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply ImageNet normalization if specified
        if self.normalize:
            image = (image - self.mean) / self.std
            
        return image
    