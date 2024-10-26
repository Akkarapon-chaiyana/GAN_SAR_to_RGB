# Paths to SAR and target image directories
sar_image_dir = 's1-s2-data/s1/'
output_dir = 'output/inference/'
import os
import re

def find_latest_model(model_dir):
    # Regular expression pattern to extract the epoch number
    pattern = re.compile(r'best_generator_epoch_(\d+)\.h5')

    # Initialize variables to track the latest model
    latest_epoch = -1
    latest_model_file = None

    # Iterate over all files in the directory
    for model_file in os.listdir(model_dir):
        match = pattern.match(model_file)
        if match:
            # Extract the epoch number from the filename
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_model_file = model_file

    return latest_model_file

# Example usage:
model_dir = './'
latest_model = find_latest_model(model_dir)

if latest_model:
    print(f"The latest model file is {latest_model}")
else:
    print("No model files found.")

h5_file_path = f"{latest_model}"

import torch
import h5py
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import datasets, transforms
from tifffile import imread
from PIL import Image
import numpy as np
import os
import warnings

# Load the trained generator model from .h5 file
def load_generator_model(h5_file_path, model):
    with h5py.File(h5_file_path, 'r') as f:
        state_dict = {}
        for key, val in f.items():
            if isinstance(val, h5py.Dataset) and np.issubdtype(val.dtype, np.number):  # Check if it's a numeric dataset
                state_dict[key] = torch.tensor(np.array(val))
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow for missing keys or mismatches
    return model

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

# 
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

def load_and_normalize_sar(file_path):
    with rasterio.open(file_path) as src:
        sar_image = src.read()  # Load the image (bands, height, width)

    # Normalize SAR image between 0 and 1
    sar_image = sar_image.astype(np.float32)
    sar_image = (sar_image - np.min(sar_image)) / (np.max(sar_image) - np.min(sar_image))
    sar_tensor = torch.from_numpy(sar_image).float().unsqueeze(0)

    return sar_tensor

def export_generated_image(generator, sar_image_path, save_path=None):
    # Load SAR image
    sar_image = load_and_normalize_sar(sar_image_path).to(device)

    # Predict the RGB image using the generator
    generator.eval()  # Set to evaluation mode
    with torch.no_grad():
        generated_rgb = generator(sar_image).cpu().squeeze(0)  # Remove batch dimension

    # Convert to NumPy format and permute dimensions to (H, W, C)
    generated_rgb_np = generated_rgb.permute(1, 2, 0).numpy()

    # Load metadata from the target image
    with rasterio.open(sar_image_path) as target_image:
        metadata = target_image.meta.copy()  # Copy metadata
        metadata.update({
            "count": 3,  # Number of bands (RGB)
            "driver": "GTiff"  # Save as GeoTIFF
        })

        # Save the generated image with the metadata
        if save_path:
            with rasterio.open(save_path, 'w', **metadata) as dst:
                # Write each band to the file
                dst.write(generated_rgb_np[:, :, 0], 1)  # Red band
                dst.write(generated_rgb_np[:, :, 1], 2)  # Green band
                dst.write(generated_rgb_np[:, :, 2], 3)  # Blue band

    # Print the shape of the array
    # print(f"Generated image shape: {generated_rgb_np.shape}")

# Load the generator model
generator = Generator().to(device)
generator = load_generator_model(h5_file_path, generator)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory at {output_dir}")

# Batch process all SAR images
for sar_image_name in os.listdir(sar_image_dir):
    if sar_image_name.endswith('.tif'):
        sar_image_path = os.path.join(sar_image_dir, sar_image_name)

        save_generated_image_path = os.path.join(output_dir, f'generated_{sar_image_name}')

        # Export the generated image with metadata from the target image
        export_generated_image(generator, sar_image_path, save_path=save_generated_image_path)
        print(f'exporting generated_{sar_image_name}')

print("Done!!!")
