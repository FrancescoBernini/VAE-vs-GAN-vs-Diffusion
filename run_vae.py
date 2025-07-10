# VAE
from common import log, get_device, parse_dataset_arg, set_seed, CustomImageDataset, get_dataloader

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio

# For reproducibility
set_seed(42) 

# Try to use GPU if available
device = get_device() 

log(f"Using the device: {device}") 

# Get dataset directory from command line argument
data_dir = parse_dataset_arg() 

# Load dataset
dataset = CustomImageDataset(data_dir, "vae")
dataloader = get_dataloader(data_dir, "vae")

log(f"Dataset with {len(dataset)} images loaded from {data_dir}.")

# Model
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

# Model loading and training setup
vae = SimpleVAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
latent_dim = 128
epochs = 50

# Training
log("Starting VAE training...")
frames = []
z_fixed = torch.randn(1, latent_dim).to(device) # Created once and used throughout training
vae.train()
for epoch in range(epochs):
    for imgs in dataloader:
        imgs = imgs.to(device)
        recons, mu, logvar = vae(imgs)
        recon_loss = nn.functional.mse_loss(recons, imgs)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.shape[0]
        loss = recon_loss + 1e-3 * kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Save the frame
    vae.eval()
    with torch.no_grad():
        vae_sample = vae.decode(z_fixed)[0].cpu() # Always use the same input
        vae_sample = vae_sample.permute(1,2,0).numpy()
        vae_sample = (vae_sample * 255).astype("uint8")
        frames.append(vae_sample)
    vae.train()
    log("Epoch {}/{} completed - Loss: {:.4f}".format(epoch+1, epochs, loss.item()))

# Save the gif and the last frame separately
plt.imsave("vae_final.png", frames[-1])
imageio.mimsave("vae_progress.gif", frames, fps=2)
log("VAE completed.")
