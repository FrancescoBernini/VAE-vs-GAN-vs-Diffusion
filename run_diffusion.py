# Diffusion

import os
import sys
import torch
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import warnings
import imageio

from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=FutureWarning)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {msg}")
    sys.stdout.flush()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Dispositivo utilizzato: {device}")

# Argomenti
if len(sys.argv) < 2:
    print("Usage: python run_diffusion.py -cats")
    sys.exit(1)

dataset_flag = sys.argv[1]
if dataset_flag == "-cats":
    data_dir = "/hpc/scratch/francesco.bernini1/AI2425/datasets/cats/Data"
else:
    raise ValueError(f"Dataset '{dataset_flag}' non gestito. Per ora supportato solo '-cats'.")

# Dataset personalizzato
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_size=64):
        self.image_paths = glob(os.path.join(image_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# Parametri
image_size = 64
batch_size = 256
num_epochs = 50

dataloader = DataLoader(
    CustomImageDataset(data_dir, image_size),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=os.cpu_count() // 2 if os.cpu_count() else 4
)
log(f"Dataset caricato con {len(dataloader.dataset)} immagini PNG. Batch size: {batch_size}")

# Importa la libreria Diffusion
try:
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion
except ImportError:
    print("La libreria 'denoising-diffusion-pytorch' non è installata.")
    sys.exit(1)

# Modello e Diffusion
model = Unet(dim=64, dim_mults=(1, 2, 4)).to(device)
diffusion = GaussianDiffusion(model, image_size=image_size, timesteps=1000).to(device)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
scaler = GradScaler()

diffusion.train()
log("Inizio training Diffusion...")

# Lista per salvare le immagini di ogni epoca
epoch_samples = []

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
    epoch_samples.append(img)
    diffusion.train()

# Salva la GIF
imageio.mimsave("diffusion_progress.gif", epoch_samples, fps=2)
log("GIF di generazione salvata come 'diffusion_progress.gif'.")

# Salva l'immagine finale
Image.fromarray(epoch_samples[-1]).save("diffusion_final.png")
log("Immagine finale salvata come 'diffusion_final.png'.")

log("Diffusion completato.")

