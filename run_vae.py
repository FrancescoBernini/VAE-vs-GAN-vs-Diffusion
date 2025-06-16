#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script di generazione immagini con VAE (completo e didattico).
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

# Funzione di log
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {msg}")
    sys.stdout.flush()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argomenti
if len(sys.argv) < 2:
    print("Usage: python run_vae.py -cats")
    sys.exit(1)

dataset_flag = sys.argv[1]
if dataset_flag == "-cats":
    data_dir = "/hpc/home/francesco.bernini1/AI2425/datasets/cats/Data"
else:
    raise ValueError("Dataset '{}' non gestito. Per ora supportato solo '-cats'.".format(dataset_flag))

# Dataset personalizzato
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_size=64):
        self.image_paths = glob(os.path.join(image_dir, "*.jpg")) + \
                           glob(os.path.join(image_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# Caricamento dati
dataset = CustomImageDataset(data_dir, image_size=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
log("Dataset caricato con {} immagini.".format(len(dataset)))

# Modello VAE
class SimpleVAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(SimpleVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*16*16, latent_dim)
        self.fc_logvar = nn.Linear(64*16*16, latent_dim)
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64*16*16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64,16,16)),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Sigmoid()
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

# Istanziamento modello
vae = SimpleVAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
num_epochs = 20

# Training
log("Inizio training VAE...")
frames = []
z = torch.randn(1,128).to(device) # Creato una volta sola
vae.train()
for epoch in range(num_epochs):
    for imgs in dataloader:
        imgs = imgs.to(device)
        recons, mu, logvar = vae(imgs)
        recon_loss = nn.functional.mse_loss(recons, imgs)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.shape[0]
        loss = recon_loss + 1e-3 * kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Salva frame
    vae.eval()
    with torch.no_grad():
        vae_sample = vae.decode(z)[0].cpu() # Usa sempre lo stesso z
        vae_sample = vae_sample.permute(1,2,0).numpy()
        vae_sample = (vae_sample * 255).astype("uint8")
        frames.append(vae_sample)
    vae.train()
    log("Epoca {}/{} completata - Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# Salva l'immagine finale e la gif
plt.imsave("vae_final.png", frames[-1])
imageio.mimsave("vae_progress.gif", frames, fps=2)
log("VAE completato.")

