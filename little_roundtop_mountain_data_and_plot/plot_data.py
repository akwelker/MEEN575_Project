import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Open the .tif file using rasterio
with rasterio.open('little_roundtop_mountain_data_and_plot\output_USGS1m.tif') as src:
    # Read the raster data into a numpy array
    array = src.read(1)
    # Define the extent of the raster
    left, bottom, right, top = src.bounds

# Plot the raster using matplotlib
fig, ax = plt.subplots()
plt.imshow(array, extent=[left, right, bottom, top], cmap='terrain')
plt.colorbar()
plt.show()

print(np.size(array))