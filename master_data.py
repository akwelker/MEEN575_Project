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


get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)   #Finds index of max value
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)   #Finds index of min value

lon, lat, elevation, transform = Extract_Geotiff_Data()

end_point_idx = get_max_index(elevation)
end_point_lat = lat[end_point_idx]
end_point_lon = lon[end_point_idx]

print(f"ending index: {end_point_idx}\n")

print(elevation[end_point_idx])
print(end_point_lon)
print(end_point_lat)

start_point_lon = -76.930
start_point_lat = -40.0975


# Create a contour plot of the topographical data
print(f"Start Location: {start_point_lon},{start_point_lat}")
print(f"End Location: {end_point_lon},{end_point_lat}")
plt.figure(1)
plt.contour(lon,lat, elevation, 25)
# plt.plot(start_loc[0],start_loc[1], "r*")
# plt.plot(end_loc[0], end_loc[1], "r*")
# plt.colorbar()

plt.show()
