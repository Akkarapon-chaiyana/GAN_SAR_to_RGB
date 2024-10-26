import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import h5py
from torchvision import datasets, transforms
from tifffile import imread
from PIL import Image
import numpy as np
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(0)

# Define dataset and dataloader
sar_dir = 's1-s2-data/s1'
rgb_dir = 's1-s2-data/s2'

# Hyperparameters
batch_size = 4
L1_lambda = 100
NUM_EPOCHS = 150
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import os
import numpy as np

def load_and_normalize_sar(file_path):
    with rasterio.open(file_path) as src:
        sar_image = src.read()  # Load the image (bands, height, width)

    # Normalize SAR image between 0 and 1
    sar_image = sar_image.astype(np.float32)
    sar_image = (sar_image - np.min(sar_image)) / (np.max(sar_image) - np.min(sar_image))

    return sar_image

def load_and_normalize_rgb(file_path):
    with rasterio.open(file_path) as src:
        rgb_image = src.read()  # Load the image (bands, height, width)

    # Normalize RGB image (assuming pixel values range up to 10000 for Sentinel-2)
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = rgb_image / 10000.0  # Normalize between 0 and 1

    return rgb_image

# Custom dataset to load both SAR and RGB
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sar_dir, rgb_dir, transform=None):
        self.sar_dir = sar_dir
        self.rgb_dir = rgb_dir
        self.transform = transform
        self.sar_files = sorted(os.listdir(sar_dir))  # Get all SAR files
        self.rgb_files = sorted(os.listdir(rgb_dir))  # Get all RGB files

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        # Load SAR and RGB images
        sar_image = load_and_normalize_sar(os.path.join(self.sar_dir, self.sar_files[idx]))
        rgb_image = load_and_normalize_rgb(os.path.join(self.rgb_dir, self.rgb_files[idx]))

        # Permute the dimensions to match the expected PyTorch format: (channels, height, width)
        sar_image = torch.from_numpy(sar_image).float()  # Convert to tensor
        rgb_image = torch.from_numpy(rgb_image).float().permute(0, 1, 2)  # Permute to (channels, height, width)

        # Optionally apply transformations (normalize, augment, etc.) if transform is defined
        if self.transform:
            sar_image = self.transform(sar_image)
            rgb_image = self.transform(rgb_image)

        return sar_image, rgb_image

# Visualizing batch
def visualize_batch(sar_image, rgb_image):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # SAR visualization: Using first channel for visualization
    ax[0].imshow(sar_image[0], cmap='gray')
    ax[0].set_title('SAR Image')

    # RGB visualization: Visualizing as usual
    ax[1].imshow(rgb_image.permute(1, 2, 0))  # Permute back to (height, width, channels) for visualization
    ax[1].set_title('RGB Image')

    plt.show()

dataset = CustomDataset(sar_dir, rgb_dir, transform=None)  # No need to use transform here
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Define DownSample block
class DownSample(nn.Module):
    def __init__(self, Input_Channels, Output_Channels):
        super(DownSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(Input_Channels, Output_Channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

# Define Upsample block
class Upsample(nn.Module):
    def __init__(self, Input_Channels, Output_Channels):
        super(Upsample, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(Input_Channels, Output_Channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(Output_Channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# Define Generator (U-Net)
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.down5 = DownSample(512, 512)
        self.down6 = DownSample(512, 512)
        self.down7 = DownSample(512, 512)
        self.down8 = DownSample(512, 512)

        self.up1 = Upsample(512, 512)
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 512)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        return u8

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define loss functions and optimizers
loss_comparison = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()
discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
generator_opt = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

def save_model_h5(model, epoch, loss, optimizer, filename='best_model.h5'):
    with h5py.File(filename, 'w') as f:
        f.attrs['epoch'] = epoch
        f.attrs['loss'] = loss.cpu().item()

        # Save model state_dict
        for key, val in model.state_dict().items():
            f.create_dataset(key, data=val.cpu().numpy())

        # Save optimizer state_dict
        for key, val in optimizer.state_dict().items():
            grp = f.create_group(f'optimizer/{key}')
            if isinstance(val, dict):
                # Handle dictionary items in optimizer state
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor):
                        grp.create_dataset(subkey, data=subval.cpu().numpy())
                    elif isinstance(subval, (int, float)):  # Save scalars directly
                        grp.attrs[subkey] = subval
            # Skip lists and other unsupported types silently


# Define training functions
def discriminator_training(inputs, targets, discriminator_opt):
    discriminator_opt.zero_grad()
    output = discriminator(inputs, targets)
    real_loss = loss_comparison(output, torch.ones_like(output, device=device))
    gen_image = generator(inputs).detach()
    fake_output = discriminator(inputs, gen_image)
    fake_loss = loss_comparison(fake_output, torch.zeros_like(fake_output, device=device))
    total_loss = (real_loss + fake_loss) / 2
    total_loss.backward()
    discriminator_opt.step()
    return total_loss

def generator_training(inputs, targets, generator_opt, L1_lambda):
    generator_opt.zero_grad()
    generated_image = generator(inputs)
    disc_output = discriminator(inputs, generated_image)
    generator_loss = loss_comparison(disc_output, torch.ones_like(disc_output, device=device))
    L1_loss_val = L1_lambda * L1_loss(generated_image, targets)
    total_loss = generator_loss + L1_loss_val
    total_loss.backward()
    generator_opt.step()
    return total_loss, generated_image

# Training loop
best_gen_loss = float('inf')

# Function for denormalizing the images for display
Normalization_Values = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def DeNormalize(tensor_of_image):
    return tensor_of_image * Normalization_Values[1][0] + Normalization_Values[0][0]

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm  # Assuming you're using tqdm for progress tracking

# Visualizing one batch
def print_images(sar_image, generated_image, rgb_image, save_path, i=0):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 3 columns

    # SAR visualization: Using first channel for visualization (assuming SAR is single channel)
    ax[0].imshow(sar_image[0].cpu().numpy(), cmap='gray')  # Visualizing the first SAR image
    ax[0].set_title('SAR Image')

    # Generated Image visualization (ensure it's detached from computation graph and converted to NumPy)
    ax[1].imshow(generated_image.transpose(1, 2, 0))  # Assuming it's already a NumPy array
    ax[1].set_title('Generated Image')

    # RGB Image visualization (assuming rgb_image is a tensor)
    ax[2].imshow(rgb_image.permute(1, 2, 0).cpu().numpy())  # Convert to numpy
    ax[2].set_title('RGB Image')

    # Save the concatenated image for side-by-side comparison
    plt.tight_layout()

    # Create a unique file name by using the epoch number, batch index, and i value
    plt.savefig(f"{save_path}/comparison_epoch_{epoch}_image_{i}.png")
    plt.show()

# Training loop
# Initialize lists to store loss values
gen_losses = []
disc_losses = []
best_gen_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    # Create a new folder for this epoch
    save_path = f"./output/epoch_{epoch+1}"
    os.makedirs(save_path, exist_ok=True)

    # Initialize variables to store losses per epoch
    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0
    batch_count = 0

    for batch_index, (images, targets) in enumerate(tqdm(dataloader)):
        inputs = images.to(device)
        targets = targets.to(device)

        # Train discriminator and accumulate the loss
        Disc_Loss = discriminator_training(inputs, targets, discriminator_opt)
        epoch_disc_loss += Disc_Loss.item()

        # Train generator multiple times and accumulate the loss
        for _ in range(10):  # Update the generator multiple times
            Gen_Loss, generator_image = generator_training(inputs, targets, generator_opt, L1_lambda)
        epoch_gen_loss += Gen_Loss.item()

        # Save the best generator model
        if Gen_Loss < best_gen_loss:
            best_gen_loss = Gen_Loss
            save_model_h5(generator, epoch+1, best_gen_loss, generator_opt, filename=f'best_generator_epoch_{epoch+1}.h5')

        # Save images for the first 2 batches of every 5th epoch
        if epoch % 5 == 0 and batch_count < 2:  # Only save for first 2 batches
            for i in range(0, 2):  # Save up to 2 images from each batch
                print_images(inputs[i].cpu(), generator_image.cpu().detach().numpy()[i], targets[i].cpu(), save_path, i=i)
            batch_count += 1  # Increment the batch count for saving images only

    # Calculate average loss per epoch
    avg_gen_loss = epoch_gen_loss / len(dataloader)
    avg_disc_loss = epoch_disc_loss / len(dataloader)

    # Append losses to the lists
    gen_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)

    print(f"Epoch {epoch+1} - Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

# Plotting losses after training is complete
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), gen_losses, label='Generator Loss')
plt.plot(range(1, NUM_EPOCHS+1), disc_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss Over Epochs')
plt.legend()
plt.grid(True)

# Save the plot as an image file (e.g., PNG or JPEG)
plot_save_path = './output/loss_plot.png'  # Define the save path
plt.savefig(plot_save_path)  # Save the plot to file
plt.close()  # Close the plot to free memory







