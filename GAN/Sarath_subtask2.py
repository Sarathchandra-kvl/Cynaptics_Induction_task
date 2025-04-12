import numpy as np 
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
from torch import device
import torch.nn as nn
import cv2
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset directory
direc = 'Enter your dataset directory here'

# Hyper parameters
image_size = 64
batch_size = 128
latent_size= 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
num_epochs=50
learning_rate=0.0002

# Transformations needed to apply to data(Data preprocessingg)
train = ImageFolder(direc, transform=tt.Compose([ tt.Resize(image_size),
                                                        tt.ToTensor(),
                                                        tt.Normalize(*stats)]))
 

#For loading data into batches
train_dl = DataLoader(train, batch_size, shuffle=True,  pin_memory=True) 

# Functions that a generator undergoes to convert random noise to image
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
 
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    )
'''
I used tanh here cause I want the pixel values to have a range -1 to 1,
as we normalized the dataset to have those values and mean to  0 and SD=1
Basically,to ensure compatibility with image training dataset

'''
# Functioning of a discriminator which takes the generated image and classifies it into true or false
discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
   
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid())
'''
Sigmoid output can be interpreted to  probability,close to 1 meaning real,else fake
and also we use binary_cross_entropy function so it expects values in range[0,1],so sigmoid is a good fit

'''

# ---------------------------------------------------------------------------------------
# The below steps are used to save the images..

# to undo the normalization process done during the preprocessing
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

#To save the samples produced during epochs

sample_dir = 'Enter directory to save the samples produced'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors).to(device)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    # If you want to see the image..
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

# ---------------------------------------------------------------------------------------

def train_discriminator(real_images, opt_d):
    
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images).to(device) 
    real_targets = torch.ones(real_images.size(0), 1).to(device) 
    real_loss = F.binary_cross_entropy(real_preds, real_targets) 
    percent_real = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1).to(device) #generating the random noises for input image
    fake_images = generator(latent).to(device)  #getting the fake images

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1).to(device) #setting target 0 
    fake_preds = discriminator(fake_images).to(device)  #getting the predictions for fake images
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)  #Comparing the two scores through loss
    percent_fake = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), percent_real, percent_fake

def train_generator(opt_g):
    
    opt_g.zero_grad()
    
    # Generate fake images
    random_noise = torch.randn(batch_size, latent_size, 1,1).to(device) #generating a random latent vector
    fake_images = generator(random_noise).to(device) 
    
    # Try to fool the discriminator
    preds = discriminator(fake_images).to(device) #getting the predictions of discriminator for fake images
    targets = torch.ones(batch_size, 1).to(device) #setting target 1 
    loss = F.binary_cross_entropy(preds, targets) 
    
    # Update generator weights--applying feedback
    loss.backward()
    opt_g.step()
    
    return loss.item(),random_noise

def run_GAN(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Create optimizers
    opt_disc = torch.optim.Adam(discriminator.to(device).parameters(), lr=lr, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(generator.to(device).parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Running test data into model
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            
            # Training arc antagonist
            real_images= real_images.to(device)
            loss_disc, real_percent, fake_percent = train_discriminator(real_images, opt_disc)
            
            # Train arc protagonist
            loss_gen, latent = train_generator(opt_gen)
            
            
        # Stats
        print(f"Epoch [{epoch+1}/{epochs}], loss_generator: {loss_gen:.6f}, loss_discriminator: {loss_disc:.6f}, percentage_real: {real_percent*100:.3f}%, percentage_fake: {fake_percent*100:.3f}%")
    
        
        save_samples(epoch+start_idx, latent, show=False)
    
 
run_GAN(epochs=num_epochs, lr=learning_rate) 

