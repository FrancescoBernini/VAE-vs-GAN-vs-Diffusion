# Diffusion
from common import log, get_device, parse_dataset_arg, set_seed, CustomImageDataset, get_dataloader

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import imageio

from torch.amp import autocast
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# For reproducibility
set_seed(42) 

# Try to use GPU if available
device = get_device() 

log(f"Using the device: {device}") 

# Get dataset directory from command line argument
data_dir = parse_dataset_arg() 

# Load dataset
dataset = CustomImageDataset(data_dir, "diffusion")
dataloader = get_dataloader(data_dir, "diffusion")

log(f"Dataset with {len(dataset)} images loaded from {data_dir}.")

# Model
class SimpleDiffusion(nn.Module):
    def __init__(self, image_size=64, timesteps=1000, dim=64):
        super().__init__()
        self.model = Unet(dim=dim, dim_mults=(1, 2, 4)) # dim_mults define the number of channels in each layer, here 64, 128, 256
        self.diffusion = GaussianDiffusion( # a bit of noise is added in each one of the 1000 timesteps 
            self.model,
            image_size=image_size,
            timesteps=timesteps
        )

    def forward(self, imgs):
        return self.diffusion(imgs)

    def sample(self, batch_size=1):
        return self.diffusion.sample(batch_size=batch_size)

# Model loading and training setup
diffusion_model = SimpleDiffusion().to(device)
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4 * 4) # x4 because 256/64=4 batch size 
best_loss = float('inf')
best_frame = None
num_epochs = 200

# Training
log("Starting Diffusion training...")
frames = []
diffusion_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, imgs in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.bfloat16): # bfloat16 better on modern gpus
            loss = diffusion_model(imgs) # loss between the U-Net noise predictions and the actual noise
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()        
    # Epoch average losses    
    avg_loss = epoch_loss / len(dataloader)

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        log(f"GPU memory allocated: {mem_alloc:.2f} MB - Reserved: {mem_reserved:.2f} MB")

    # Save the frame
    diffusion_model.eval()
    with torch.no_grad():
        set_seed(42) # Ensure same starting noise every time
        sampled_image = diffusion_model.sample(batch_size=1)[0]
        
    img = sampled_image.permute(1,2,0).cpu().numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    frames.append(img)
    
    # Save best frame
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_frame = img.copy()    
    diffusion_model.train()
    log("Epoch {}/{} completed - Average Loss: {:.4f}".format(epoch+1, num_epochs, avg_loss))


# Save the gif and the best frame separately
plt.imsave("diffusion_best.png", best_frame)
imageio.mimsave("diffusion_progress.gif", frames, fps=2)
log("Diffusion completed.")

