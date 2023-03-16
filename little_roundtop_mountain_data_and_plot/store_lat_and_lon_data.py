import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Open the .tif file using rasterio
with rasterio.open('little_roundtop_mountain_data_and_plot\output_USGS1m.tif') as src:
    # Read the raster data into a numpy array
    array = src.read(1)
    # Define the extent of the raster
    left, bottom, right, top = src.bounds
    # Get the affine transform of the raster
    transform = src.transform

# Create meshgrid of coordinates
cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
# Transform coordinates to latitude and longitude
lon, lat = transform * (cols, rows)
# Get elevation data
elevation = array

# Plot the raster using matplotlib
fig, ax = plt.subplots()
plt.imshow(array, extent=[left, right, bottom, top], cmap='terrain')
plt.colorbar()
plt.show()

print(lon)
print(np.shape(lon))
print(np.shape(lat))
print(np.shape(elevation))
