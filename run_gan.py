#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script di generazione immagini con GAN (completo e didattico).
"""
import os
import sys
import torch
import datetime
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {msg}")
    sys.stdout.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 2:
    print("Usage: python run_gan.py -cats")
    sys.exit(1)

dataset_flag = sys.argv[1]
if dataset_flag == "-cats":
    data_dir = "/hpc/home/francesco.bernini1/AI2425/datasets/cats/Data"
else:
    raise ValueError("Dataset '{}' non gestito. Per ora supportato solo '-cats'.".format(dataset_flag))

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_size=64):
        self.image_paths = glob(os.path.join(image_dir, "*.jpg")) + \
                           glob(os.path.join(image_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

dataset = CustomImageDataset(data_dir, image_size=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

log("Dataset caricato con {} immagini.".format(len(dataset)))

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class SimpleDiscriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(SimpleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*16*16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

generator = SimpleGenerator().to(device)
discriminator = SimpleDiscriminator().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()
latent_dim = 100
epochs = 20

log("Inizio training GAN...")
frames = []

best_loss_g = float('inf')
best_frame = None

z_fixed = torch.randn(1, latent_dim, 1, 1, device=device)
generator.train()
discriminator.train()
for epoch in range(epochs):
    for imgs in dataloader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        validity = discriminator(gen_imgs)
        g_loss = loss_fn(validity, valid)
        g_loss.backward()
        optimizer_G.step()
        # Train Discriminator
        optimizer_D.zero_grad()
        real_validity = discriminator(imgs)
        d_real_loss = loss_fn(real_validity, valid)
        fake_validity = discriminator(gen_imgs.detach())
        d_fake_loss = loss_fn(fake_validity, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    # Salva frame
    generator.eval()
    with torch.no_grad():
        gen_sample = generator(z_fixed).cpu()
        gen_sample = (gen_sample + 1) / 2
        gen_sample = gen_sample[0].permute(1,2,0).numpy()
        gen_sample = (gen_sample * 255).astype("uint8")
        frames.append(gen_sample)
        # Salva la migliore generazione
        if g_loss.item() < best_loss_g:
            best_loss_g = g_loss.item()
            best_frame = gen_sample.copy()
    generator.train()
    log("Epoca {}/{} completata - Loss D: {:.4f}, Loss G: {:.4f}".format(epoch+1, epochs, d_loss.item(), g_loss.item()))

# Salva la GIF e l'immagine migliore
plt.imsave("gan_best.png", best_frame)
imageio.mimsave("gan_progress.gif", frames, fps=2)
log("GAN completato.")

