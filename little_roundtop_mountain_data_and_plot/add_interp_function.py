import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Define the bounding box of the target geographical range
minX, minY = (-76.943693, 40.090454)
maxX, maxY = (-76.907129, 40.115401)

def Extract_Geotiff_Data():

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
    return lon, lat, elevation, transform

lon, lat, elevation, transform = Extract_Geotiff_Data()

# Define a function to estimate elevation at a given point
def Get_Elevation(lat, lon):
    # Map the input latitude and longitude to the indices of the elevation array
    i = int(np.interp(lat, (minY, maxY), (elevation.shape[0]-1, 0)))
    j = int(np.interp(lon, (minX, maxX), (0, elevation.shape[1]-1)))
    # Use bilinear interpolation to estimate the elevation at the given point
    x1, x2 = lon, lon+transform[0]
    y1, y2 = lat, lat+transform[4]
    q11, q21 = elevation[i,j], elevation[i,j+1]
    q12, q22 = elevation[i+1,j], elevation[i+1,j+1]
    elevation_at_point = ((q11 * (x2 - lon) * (y2 - lat) + 
                            q21 * (lon - x1) * (y2 - lat) + 
                            q12 * (x2 - lon) * (lat - y1) + 
                            q22 * (lon - x1) * (lat - y1)) / ((x2 - x1) * (y2 - y1)))
    return elevation_at_point

# Create a contour plot of the topographical data
fig, ax = plt.subplots()
contour = ax.contour(lon, lat, elevation, levels=20, cmap='terrain')
ax.clabel(contour, inline=True, fontsize=8)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Contour Plot of Topographical Data')

# Plot the raster using matplotlib
fig, ax = plt.subplots()
plt.imshow(elevation, extent=[minX, maxX, minY, maxY], cmap='terrain')
plt.colorbar()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Topographical Data')

plt.show()


max_elevation = np.amax(elevation)
print(max_elevation)
[lat_max, lon_max] = np.where(elevation == max_elevation)
print(np.shape(lat))
print(np.shape(lon))
# print(f"Elevation at ({point_lat}, {point_lon}): {max_elevation:.2f} meters")


# # Test the function by estimating the elevation at a given point
# point_lat, point_lon = 40.100, -76.930
# point_elevation = get_elevation(point_lat, point_lon)
# print(f"Elevation at ({point_lat}, {point_lon}): {point_elevation:.2f} meters")

# # Plot the estimated elevation point on the map
# plt.scatter(point_lon, point_lat, color='red', marker='x')
# plt.show()