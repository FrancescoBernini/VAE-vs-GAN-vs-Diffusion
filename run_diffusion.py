# Diffusion
from common import log, get_device, parse_dataset_arg, set_seed, CustomImageDataset, get_dataloader

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio

from torch.cuda.amp import autocast, GradScaler

# Importa la libreria Diffusion
try:
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion
except ImportError:
    print("La libreria 'denoising-diffusion-pytorch' non è installata.")
    sys.exit(1)

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

# Model loading and training setup
model = Unet(dim=64, dim_mults=(1, 2, 4)).to(device)
diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000).to(device)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
scaler = GradScaler()
num_epochs = 30

# Training
log("Starting Diffusion training...")
frames = []
diffusion.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, imgs in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            loss = diffusion(imgs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        if (i + 1) % 100 == 0:
            log(f"Epoca {epoch+1}/{num_epochs} - Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)
    log(f"Epoca {epoch+1}/{num_epochs} completata - Loss media: {avg_loss:.4f}")

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        log(f"Memoria GPU allocata: {mem_alloc:.2f} MB - Riservata: {mem_reserved:.2f} MB")

    # Genera un'immagine alla fine di ogni epoca per la GIF
    diffusion.eval()
    with torch.no_grad():
        sampled_image = diffusion.sample(batch_size=1)[0]
    img = sampled_image.permute(1,2,0).cpu().numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    frames.append(img)
    diffusion.train()

# Save the gif and the best frame separately
plt.imsave("diffusion_final.png", frames[-1])
imageio.mimsave("diffusion_progress.gif", frames, fps=2)
log("Diffusion completed.")

