import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Define the bounding box of the target geographical range
minX, minY = (-76.943693, 40.090454)
maxX, maxY = (-76.907129, 40.115401)

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

# Map latitude and longitude to the target bounding box
lon = np.interp(lon, (left, right), (minX, maxX))
lat = np.interp(lat, (bottom, top), (minY, maxY))

# Get elevation data
elevation = array

# Plot the raster using matplotlib
plt.imshow(array, extent=[minX, maxX, minY, maxY], cmap='terrain')
plt.colorbar()
plt.show()