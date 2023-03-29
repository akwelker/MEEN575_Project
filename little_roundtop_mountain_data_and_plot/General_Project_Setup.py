import rasterio
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import minimize, NonlinearConstraint, Bounds 
from scipy.interpolate import RectBivariateSpline

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
    x_data_grid, y_data_grid = transform * (cols, rows)

    # Map latitude and longitude to the target bounding box
    x_data_grid = np.interp(x_data_grid, (left, right), (minX, maxX))
    y_data_grid = np.interp(y_data_grid, (bottom, top), (minY, maxY))

    x_data = x_data_grid[0,:]
    y_data = np.flip(y_data_grid[:,0])
    print('ydata point', y_data[0])
    print('xdata', x_data)
    print('ydata', y_data)
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

def toblers_hiking_function(slope):         
    return 1000*6*math.exp(-3.5*abs(slope+.05))    #Computes speed in m/hr

def objective_f(x):

    segment_topview_distance = np.zeros(num_segments)
    segment_true_distance = np.zeros(num_segments)
    segment_slope = np.zeros(num_segments)
    segment_speed = np.zeros(num_segments)
    segment_angle = np.zeros(num_segments)
    points_elevation = np.zeros(num_segments+1)

    path_x_points = np.zeros(num_total_points)
    path_y_points = np.zeros(num_total_points)
    path_x_points[0] = start_point_x
    path_y_points[0] = start_point_y
    path_x_points[-1] = end_point_x
    path_y_points[-1] = end_point_y

    for i in range(num_waypoints):
        path_x_points[i+1] = x[2*i]
        path_y_points[i+1] = x[2*i+1]

    for i in range(num_total_points):
        points_elevation[i] = Get_Elevation(path_x_points[i],path_y_points[i])
    
    for i in range(num_waypoints + 1):
        segment_topview_distance[i] = math.sqrt((path_x_points[i] - path_x_points[i+1])**2 + (path_y_points[i]-path_y_points[i+1])**2)
        segment_true_distance[i] = math.sqrt((points_elevation[i+1]-points_elevation[i])**2 + segment_topview_distance[i]**2)
        segment_slope[i] = (points_elevation[i+1]-points_elevation[i])/segment_topview_distance[i]
        segment_speed[i] = toblers_hiking_function(segment_slope[i])
        segment_angle[i] = math.degrees(math.atan(segment_slope[i]))

    time_to_ascend = np.dot(segment_true_distance, np.reciprocal(segment_speed)) * 60       #In minutes
    # print('time', time_to_ascend)

    return time_to_ascend

def g1(x):

    segment_topview_distance = np.zeros(num_segments)
    segment_slope = np.zeros(num_segments)
    segment_angle = np.zeros(num_segments)
    points_elevation = np.zeros(num_segments+1)

    path_x_points = np.zeros(num_total_points)
    path_y_points = np.zeros(num_total_points)
    path_x_points[0] = start_point_x
    path_y_points[0] = start_point_y
    path_x_points[-1] = end_point_x
    path_y_points[-1] = end_point_y

    for i in range(num_waypoints):
        path_x_points[i+1] = x[2*i]
        path_y_points[i+1] = x[2*i+1]

    for i in range(num_total_points):
        points_elevation[i] = Get_Elevation(path_x_points[i],path_y_points[i])
    
    for i in range(num_waypoints + 1):
        segment_topview_distance[i] = math.sqrt((path_x_points[i] - path_x_points[i+1])**2 + (path_y_points[i]-path_y_points[i+1])**2)
        segment_slope[i] = (points_elevation[i+1]-points_elevation[i])/segment_topview_distance[i]
        segment_angle[i] = math.degrees(math.atan(segment_slope[i]))

    # print('angles', segment_angle)

    return segment_angle - slope_constraint_degrees


def dg(x):
    # compute Jg
    h = 10e-3
    nx = x.size
    ng = num_segments
    Jg = np.zeros((ng,nx))
    # g_0 = g2(x)     #Only use the slope constraints

    for j in range(0,nx):
        delta_x = h*(1+np.abs(x[j]))
        x[j]= x[j] + delta_x
        g_plus = g1(x)
        x[j]= x[j] - 2*delta_x
        g_minus = g1(x)
        Jg[:,j] = (g_plus - g_minus)/(2*delta_x)
        x[j]= x[j] + delta_x

    return Jg

def df(x):
    f_0 = objective_f(x)
    h = 10e-4
    Jf = np.zeros(np.size(x))

    for j in range(0,np.size(x)):
        delta_x = h*(1+np.abs(x[j]))
        x[j]= x[j] + delta_x
        f_plus = objective_f(x)
        x[j]= x[j] - 2*delta_x
        f_minus = objective_f(x)
        Jf[j] = (f_plus - f_minus)/(2*delta_x)
        x[j]= x[j] + delta_x

    return Jf
    
def obj(x):
    return objective_f(x), df(x)

def jac_constraint(x):
    return dg(x)

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

# print('Mins and Maxes x/y', minX, maxX, minY, maxY)

x_data, y_data, elevation, transform = Extract_Geotiff_Data()
y_data = y_data - y_data[0]

maxX = x_data[len(x_data) - 1]  #Ensuer that the interpolation onto the meter grid hasn't effected our maxes (out-of-bounds protection)
maxY = y_data[len(y_data) - 1]

interp_func = RectBivariateSpline(x_data, y_data, elevation)

end_point_idx = get_max_index(elevation)
end_point_y = y_data[end_point_idx[1]]
end_point_x = x_data[end_point_idx[0]]

# Manually Chosen Start Point
start_point_x = 2000
start_point_y = 500

num_waypoints = 25
num_segments = num_waypoints + 1
num_total_points = num_waypoints + 2

x0 = np.zeros(num_waypoints*2) 
spacing_x = (end_point_x - start_point_x)/(num_waypoints+1)
spacing_y = (end_point_y - start_point_y)/(num_waypoints+1)

range_noise = 75

for i in range(num_waypoints):
    x0[2*i] = (start_point_x + (i+1)*spacing_x) + random.uniform(-1*range_noise, range_noise)
    x0[2*i + 1] = (start_point_y + (i+1)*spacing_y) + random.uniform(-1*range_noise, range_noise)

# print('x0', x0)
print('initial time', objective_f(x0))

optimizer_options = {'disp': False, 'maxiter':1000}

from scipy.optimize import Bounds

# Define the bounds for each variable, even numbered indices (x_points) get xBounds, odd numbered indices (y_points) get yBounds
lower_bounds = [minX if i % 2 == 0 else minY for i in range(x0.size)]
upper_bounds = [maxX if i % 2 == 0 else maxY for i in range(x0.size)]

geographic_bounds = Bounds(lower_bounds, upper_bounds)

slope_constraint_degrees = 12
# segment_slope_constraints = [{'type':'ineq', 'fun':g1 }]
segment_slope_constraints = NonlinearConstraint(g1, lb = -np.inf, ub = 0, jac = jac_constraint)

# res = minimize(objective_f, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
# res = minimize(objective_f, x0, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
# res = minimize(obj, x0, bounds=geographic_bounds, tol=1e-7, options=optimizer_options, jac = True)


res = minimize(obj, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options, jac = True)
print(res)

# print('x*', res.x)
x_star = res.x

plt.contour((x_data), (y_data), np.transpose(elevation), 25)
plt.colorbar()

x_points = np.zeros(num_waypoints+2)
y_points = np.zeros(num_waypoints+2)
x_points[0] = start_point_x
y_points[0] = start_point_y
x_points[-1] = end_point_x
y_points[-1] = end_point_y

x_stars = np.zeros(num_waypoints+2)
y_stars = np.zeros(num_waypoints+2)
x_stars[0] = start_point_x
y_stars[0] = start_point_y
x_stars[-1] = end_point_x
y_stars[-1] = end_point_y

for i in range(num_waypoints):
    x_points[i+1] = x0[2*i]
    y_points[i+1] = x0[2*i+1]
    x_stars[i+1] = x_star[2*i]
    y_stars[i+1] = x_star[2*i+1]


# Create a scatter plot of the points
plt.scatter(x_points, y_points, color = 'red')

# Connect the points with lines
plt.plot(x_points, y_points, color='red')

# Create a scatter plot of the points
plt.scatter(x_stars, y_stars, color = 'blue')

# Connect the points with lines
plt.plot(x_stars, y_stars, color='blue')

# Add axis labels and a title
plt.xlabel('X Vals (meters)')
plt.ylabel('Y Vals (meters)')
plt.title('Contour plot with points')

# Show the plot
plt.show()