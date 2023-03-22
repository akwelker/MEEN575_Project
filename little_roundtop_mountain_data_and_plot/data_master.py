import rasterio
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint

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
    x_data, y_data = transform * (cols, rows)

    # Map latitude and longitude to the target bounding box
    x_data = np.interp(x_data, (left, right), (minX, maxX))
    y_data = np.interp(y_data, (bottom, top), (minY, maxY))

    # Get elevation data
    elevation = array
    elevation = elevation
    return x_data, y_data, elevation, transform

def Get_Elevation(x_point, y_point):
    # Map the input latitude and longitude to the indices of the elevation array
    i = int(np.interp(y_point, (minY, maxY), (elevation.shape[0]-1, 0)))
    j = int(np.interp(x_point, (minX, maxX), (0, elevation.shape[1]-1)))
    # Use bilinear interpolation to estimate the elevation at the given point
    x1, x2 = x_point, x_point+transform[0]
    y1, y2 = y_point, y_point+transform[4]
    q11, q21 = elevation[i,j], elevation[i,j+1]
    q12, q22 = elevation[i+1,j], elevation[i+1,j+1]
    elevation_at_point = ((q11 * (x2 - x_point) * (y2 - y_point) + 
                            q21 * (x_point - x1) * (y2 - y_point) + 
                            q12 * (x2 - x_point) * (y_point - y1) + 
                            q22 * (x_point - x1) * (y_point - y1)) / ((x2 - x1) * (y2 - y1)))
    return elevation_at_point

get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)   #Finds index of max value
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)   #Finds index of min value

def toblers_hiking_function(slope):         
    return 1000*6*math.exp(-3.5*abs(slope+.05))    #Computes speed in m/hr

def objective_f(x):
    
    way_x = x[0]
    way_y = x[1]
    topview_distance_segment_1 = math.sqrt((way_x - start_point_x)**2 + (way_y - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x - end_point_x)**2 + (way_y - end_point_y)**2)

    # print(f"distance_segment_1: {topview_distance_segment_1}\n")
    # print(f"distance_segment_2: {topview_distance_segment_2}\n")

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_mid_point = Get_Elevation(x[0],x[1])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    # print(f"elevation_start_point: {elevation_start_point}\n")
    # print(f"elevation_1: {elevation_midpoint}\n")
    # print(f"elevation_end_point: {elevation_end_point}\n")

    true_distance_segment_1 = math.sqrt((elevation_mid_point - elevation_start_point)**2 + topview_distance_segment_1**2)
    true_distance_segment_2 = math.sqrt((elevation_mid_point - elevation_end_point)**2 + topview_distance_segment_2**2)

    slope_segment_1 = math.tan((elevation_mid_point - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_end_point- elevation_mid_point)/topview_distance_segment_2)

    # print(f"slope_segment_1: {slope_segment_1}\n")
    # print(f"slope_segment_2: {slope_segment_2}\n")

    print(f"angle_segment_1: {math.degrees(math.atan(slope_segment_1))}\n")
    print(f"angle_segment_2: {math.degrees(math.atan(slope_segment_2))}\n")

    speed_segment_1 = toblers_hiking_function(slope_segment_1)
    speed_segment_2 = toblers_hiking_function(slope_segment_2)

    # print(f"speed_segment_1: {speed_segment_1}\n")
    # print(f"speed_segment_2: {speed_segment_2}\n")

    time_to_ascend = true_distance_segment_1/speed_segment_1 + true_distance_segment_2/speed_segment_2

    time_to_ascend = time_to_ascend * 60 # In minutes

    # print(f"time_to_ascend: {time_to_ascend}\n")

    print('currentX',x)
    print('Ascenion Time',time_to_ascend)

    return time_to_ascend 

def g3(x):

    x_waypoint = x[0]
    y_waypoint = x[1]
    topview_distance_segment_1 = math.sqrt((x_waypoint - start_point_x)**2 + (y_waypoint - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((x_waypoint - end_point_x)**2 + (y_waypoint - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_midpoint = Get_Elevation(x[0],x[1])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    true_distance_segment_1 = math.sqrt((elevation_midpoint - elevation_start_point)**2 + topview_distance_segment_1**2)
    true_distance_segment_2 = math.sqrt((elevation_midpoint - elevation_end_point)**2 + topview_distance_segment_2**2)

    return abs(topview_distance_segment_1 - topview_distance_segment_2) - 50

def g1(x): 
    x_waypoint = x[0]
    y_waypoint = x[1]
    topview_distance_segment_1 = math.sqrt((x_waypoint - start_point_x)**2 + (y_waypoint - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((x_waypoint - end_point_x)**2 + (y_waypoint - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_midpoint = Get_Elevation(x[0],x[1])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    true_distance_segment_1 = math.sqrt((elevation_midpoint - elevation_start_point)**2 + topview_distance_segment_1**2)
    true_distance_segment_2 = math.sqrt((elevation_midpoint - elevation_end_point)**2 + topview_distance_segment_2**2)

    slope_segment_1 = math.tan((elevation_midpoint - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_end_point- elevation_midpoint)/topview_distance_segment_2)

    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))

    return angle_segment_1 - 15

def g2(x): 
    x_waypoint = x[0]
    y_waypoint = x[1]
    topview_distance_segment_1 = math.sqrt((x_waypoint - start_point_x)**2 + (y_waypoint - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((x_waypoint - end_point_x)**2 + (y_waypoint - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_midpoint = Get_Elevation(x[0],x[1])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    true_distance_segment_1 = math.sqrt((elevation_midpoint - elevation_start_point)**2 + topview_distance_segment_1**2)
    true_distance_segment_2 = math.sqrt((elevation_midpoint - elevation_end_point)**2 + topview_distance_segment_2**2)

    slope_segment_1 = math.tan((elevation_midpoint - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_end_point- elevation_midpoint)/topview_distance_segment_2)

    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))


    return angle_segment_2 - 15

CONVERT_COORD_TO_METER = 111139

# Define the bounding box of the target geographical range
# minX, minY = (-76.943693, 40.090454)
# maxX, maxY = (-76.907129, 40.115401)

minX = 0
maxX = abs(76.907129 - 76.943693) * CONVERT_COORD_TO_METER
minY = 0
maxY = abs(40.090454 - 40.115401) * CONVERT_COORD_TO_METER


x_data, y_data, elevation, transform = Extract_Geotiff_Data()

end_point_idx = get_max_index(elevation)
end_point_y = y_data[end_point_idx]
end_point_x = x_data[end_point_idx]

# print(f"ending index: {end_point_idx}\n")

# print(elevation[end_point_idx])
# print(end_point_x)
# print(end_point_y)

# Manually Chosen Start Point
start_point_x = 500
start_point_y = 500

### START PROBLEM 4_2 ###

### THESE ARE OUR DESIGN VARIABLES ###
waypoint_x = (start_point_x + end_point_x)/2
waypoint_y = (start_point_y + end_point_y)/2
### END OF DESING VARIABLES ###

x0 = [waypoint_x, waypoint_y]

print('x0', x0)

optimizer_options = {'disp': True, 'maxiter':500}

geographic_bounds = ((minX,maxX), (minY,maxY))

def step_size_constraint(x):
    step_sizes = np.abs(x - step_size_constraint.last_x)
    return np.min(step_sizes - 1)

step_size_constraint.last_x = x0

segment_slope_constraints = [{'type':'ineq','fun':g3}]
# segment_slope_constraints = [{'type':'ineq', 'fun':g1 },{'type':'ineq', 'fun':g2},{'type':'eq','fun':g3}]
# segment_slope_constraints = [{'type':'ineq', 'fun':g1 },{'type':'ineq', 'fun':g2},{'type':'eq','fun':g3},{'type': 'ineq', 'fun': step_size_constraint}]
# Set constraint bounds to signify greater than or equal to constraints
# lg = -50
# ug = 60
# segment_slope_constraints = NonlinearConstraint(g, lg, ug)

print('x0', x0)
res = minimize(objective_f, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
# res = minimize(objective_f, x0, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
print(res)

print('x*', res.x)

x_star = res.x

# print(objective_f(x0))
# print(elevation[1000][1000])

plt.contour(x_data, y_data, elevation, 25)
plt.colorbar()

x_points = [start_point_x, waypoint_x, end_point_x]
y_points = [start_point_y, waypoint_y, end_point_y]

# Create a scatter plot of the points
plt.scatter(x_points, y_points, color = 'red')

# Connect the points with lines
plt.plot(x_points, y_points, color='red')

plt.plot(x_star[0],x_star[1],'b*')


# Add axis labels and a title
plt.xlabel('X Vals (meters)')
plt.ylabel('Y Vals (meters)')
plt.title('Contour plot with points')

# Show the plot
plt.show()