
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from generator_network import Generator,Discriminator
 
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_size = 100
batch_size = 10

train_data = training_data()

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
 )

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10


generator = Generator(latent_dim)
discriminator = Discriminator()
 
# Loss function
adversarial_loss = nn.BCELoss()
 
# Optimizers
optimizer_G = optim.Adam(generator.parameters()
                         , lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters()
                         , lr=lr, betas=(beta1, beta2))


for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
       # Convert list to tensor
        real_images = batch[0].to(device)
 
        # Adversarial ground truths
        valid = torch.ones(real_images.size(0), 1)
        fake = torch.zeros(real_images.size(0), 1)
 
        # Configure input
        real_images = real_images
 
        # ---------------------
        #  Train Discriminator
        # ---------------------
 
        optimizer_D.zero_grad()
 
        # Sample noise as generator input
        z = torch.randn(real_images.size(0), latent_dim, device=device)
 
        # Generate a batch of images
        fake_images = generator(z)
 
        # Measure discriminator's ability
        # to classify real and fake images
        real_loss = adversarial_loss(discriminator\
                                     (real_images), valid)
        fake_loss = adversarial_loss(discriminator\
                                     (fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
 
        # Backward pass and optimize
        d_loss.backward()
        optimizer_D.step()
 
        # -----------------
        #  Train Generator
        # -----------------
 
        optimizer_G.zero_grad()
 
        # Generate a batch of images
        gen_images = generator(z)
 
        # Adversarial loss
        g_loss = adversarial_loss(discriminator(gen_images), valid)
 
        # Backward pass and optimize
        g_loss.backward()
        optimizer_G.step()
 
        # ---------------------
        #  Progress Monitoring
        # ---------------------
 
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\
                        Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
 
    # Save generated images for every epoch
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated,\
                                        nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()
