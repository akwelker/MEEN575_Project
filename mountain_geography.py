import rasterio
import numpy as np
from scipy.interpolate import RectBivariateSpline


CONVERT_COORD_TO_METER = 111139

#From GEOTIFF Download
MIN_LAT = 40.090454
MIN_LON = 76.907129
MAX_LAT = 40.115401
MAX_LON = 76.943693

minX = 0
maxX = abs(MAX_LON - MIN_LON) * CONVERT_COORD_TO_METER
minY = 0
maxY = abs(MAX_LAT - MIN_LAT) * CONVERT_COORD_TO_METER

##============================= Extracting GEO DATA ============================
def Extract_Geotiff_Raw():
    # Open the .tif file using rasterio
    with rasterio.open('output_USGS1m.tif') as src:
        # Read the raster data into a numpy array
        ell_data = src.read(1)
        # Define the extent of the raster
        left, bottom, right, top = src.bounds
        # Get the affine transform of the raster
        transform = src.transform

    # Create meshgrid of coordinates
    cols, rows = np.meshgrid(np.arange(ell_data.shape[1]), np.arange(ell_data.shape[0]))
    # Transform coordinates to latitude and longitude
    x_data_grid, y_data_grid = transform * (cols, rows)

    # Map latitude and longitude to the target bounding box
    x_data_grid = np.interp(x_data_grid, (left, right), (minX, maxX))
    y_data_grid = np.interp(y_data_grid, (bottom, top), (minY, maxY))

    return x_data_grid, y_data_grid, ell_data




def Extract_Geotiff_Data():
    # Open the .tif file using rasterio
    with rasterio.open('output_USGS1m.tif') as src:
        # Read the raster data into a numpy array
        array = src.read(1)
        # Define the extent of the raster
        left, bottom, right, top = src.bounds
        # Get the affine transform of the raster
        transform = src.transform

    # Create meshgrid of coordinates
    cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    # Transform coordinates to latitude and longitude
    x_data_grid, y_data_grid = transform * (cols, rows)

    # Map latitude and longitude to the target bounding box
    x_data_grid = np.interp(x_data_grid, (left, right), (minX, maxX))
    y_data_grid = np.interp(y_data_grid, (bottom, top), (minY, maxY))

    x_data = x_data_grid[0,:]
    y_data = np.flip(y_data_grid[:,0])
    # print('ydata point', y_data[0])
    # print('xdata', x_data)
    # print('ydata', y_data)
    # Get elevation data
    elevation = np.transpose(array)
    elevation = np.flip(elevation, axis=1)

    return x_data, y_data, elevation, transform




def Get_Elevation(x_point, y_point):
    point = (x_point,y_point)
    elevation_at_point = interp_func(*point)
    return elevation_at_point

get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)   #Finds index of max value
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)   #Finds index of min value

x_data, y_data, elevation, transform = Extract_Geotiff_Data()
y_data = y_data - y_data[0]

maxX = x_data[len(x_data) - 1]  #Ensuer that the interpolation onto the meter grid hasn't effected our maxes (out-of-bounds protection)
maxY = y_data[len(y_data) - 1]

end_point_idx = get_max_index(elevation)
end_point_y = y_data[end_point_idx[1]]
end_point_x = x_data[end_point_idx[0]]

# Manually Chosen Start Point
start_point_x = 500
start_point_y = 500

num_waypoints = 12
num_segments = num_waypoints + 1
num_total_points = num_waypoints + 2

x0 = np.zeros(num_waypoints*2) 
spacing_x = (end_point_x - start_point_x)/(num_waypoints+1)
spacing_y = (end_point_y - start_point_y)/(num_waypoints+1)

interp_func = RectBivariateSpline(x_data, y_data, elevation)


