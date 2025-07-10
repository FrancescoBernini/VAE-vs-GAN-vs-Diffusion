import os
import sys
import torch
import datetime
import warnings
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore", category=FutureWarning)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {msg}")
    sys.stdout.flush()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_dataset_arg():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} -[dataset_flag]")
        sys.exit(1)

    dataset_flag = sys.argv[1]
    if dataset_flag == "-cats":
        return "/hpc/scratch/francesco.bernini1/AI2425/datasets/cats/Data"
    else:
        raise ValueError(f"Dataset '{dataset_flag}' does not exist.")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, model_type):
        self.image_paths = glob(os.path.join(image_dir, "*.png"))
        if model_type == "vae":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ]) 
        elif model_type == "gan":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) # Every channel will be normalized to [0, 1] and then to [-1, 1] for tanh compatibility

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

def get_dataloader(data_dir, model_type):
    dataset = CustomImageDataset(data_dir, model_type)
    if model_type == "vae":
        return DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4
        )
    elif model_type == "gan":
        return DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4
        )
    

def get_fixed_noise(latent_dim=128, seed=42, device=None):
    torch.manual_seed(seed)
    return torch.randn(1, latent_dim, device=device)
