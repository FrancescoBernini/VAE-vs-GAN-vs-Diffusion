# GAN
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
dataset = CustomImageDataset(data_dir, "gan")
dataloader = get_dataloader(data_dir, "gan")

log(f"Dataset with {len(dataset)} images loaded from {data_dir}.")

# Generator model
class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=128):
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

# Discriminator model
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

# Model loading and training setup
generator = SimpleGenerator().to(device)
discriminator = SimpleDiscriminator().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()
best_loss_g = float('inf')
best_frame = None
latent_dim = 128
epochs = 200

# Training
log("Starting GAN training...")
frames = []
z_fixed = torch.randn(1, latent_dim, 1, 1, device=device)
generator.train()
discriminator.train()
for epoch in range(epochs):
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    for imgs in dataloader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        real_labels_smoothed = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels_smoothed = torch.full((batch_size, 1), 0.1, device=device)
        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        validity = discriminator(gen_imgs)
        g_loss = loss_fn(validity, real_labels_smoothed)
        g_loss.backward()
        optimizer_G.step()        
        # Train Discriminator
        optimizer_D.zero_grad()        
        noise = torch.randn_like(imgs) * 0.005 
        imgs_noisy = imgs + noise # Adding noise to real images
        imgs_noisy = torch.clamp(imgs_noisy, -1.0, 1.0) 
        real_validity = discriminator(imgs_noisy)      
        d_real_loss = loss_fn(real_validity, real_labels_smoothed)
        fake_validity = discriminator(gen_imgs.detach())
        d_fake_loss = loss_fn(fake_validity, fake_labels_smoothed)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        epoch_loss_g += g_loss.item()
        epoch_loss_d += d_loss.item()
    # Epoch average losses
    avg_loss_g = epoch_loss_g / len(dataloader)
    avg_loss_d = epoch_loss_d / len(dataloader)

    # Save the frame
    generator.eval()
    with torch.no_grad():
        gen_sample = generator(z_fixed).cpu()
        gen_sample = (gen_sample + 1) / 2
        gen_sample = gen_sample[0].permute(1,2,0).numpy()
        gen_sample = (gen_sample * 255).astype("uint8")
        frames.append(gen_sample)
        # Save best frame
        if avg_loss_g < best_loss_g:
            best_loss_g = avg_loss_g
            best_frame = gen_sample.copy()
    generator.train()
    log("Epoch {}/{} completed - Avg Loss D: {:.4f}, Avg Loss G: {:.4f}".format(epoch+1, epochs, avg_loss_d, avg_loss_g))

# Save the gif and the best frame separately
plt.imsave("gan_best.png", best_frame)
imageio.mimsave("gan_progress.gif", frames, fps=2)
log("GAN completed.")

