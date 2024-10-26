import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob

# Define the directory containing your TIFF images
inference_dir = 'output/inference/'
output_dir = 'output/mosaic_output/'
output_path = os.path.join(output_dir, 'mosaic_inference.tif')  # Complete file path for saving the mosaic
print(output_path)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory at {output_dir}")

# Get all the TIFF files in the directory
tiff_files = glob.glob(os.path.join(inference_dir, '*.tif'))

# Open all TIFF files as datasets
datasets = []
for tiff in tiff_files:
    src = rasterio.open(tiff)
    datasets.append(src)
print(datasets[0:2])
# Merge all TIFF files into a single mosaic
mosaic, transform = merge(datasets)

# Get the metadata of the first file and update it for the mosaic
out_meta = datasets[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform
})

# Save the mosaic as a new TIFF file
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f'Mosaic saved to {output_path}')
