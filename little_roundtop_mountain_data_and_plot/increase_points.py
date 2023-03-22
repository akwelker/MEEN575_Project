import rasterio
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint
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
    # # Map the input latitude and longitude to the indices of the elevation array
    # i = int(np.interp(y_point, (minY, maxY), (elevation.shape[0]-1, 0)))
    # j = int(np.interp(x_point, (minX, maxX), (0, elevation.shape[1]-1)))
    # # Use bilinear interpolation to estimate the elevation at the given point
    # x1, x2 = x_point, x_point+transform[0]
    # y1, y2 = y_point, y_point+transform[4]
    # if i == (elevation.shape[0]-1) or j == (elevation.shape[1]-1):
    #     q11, q21 = elevation[i,j], elevation[i,j]
    #     q12, q22 = elevation[i,j], elevation[i,j]
    # else:
    #     q11, q21 = elevation[i,j], elevation[i,j+1]
    #     q12, q22 = elevation[i+1,j], elevation[i+1,j+1]
    # elevation_at_point = ((q11 * (x2 - x_point) * (y2 - y_point) + 
    #                         q21 * (x_point - x1) * (y2 - y_point) + 
    #                         q12 * (x2 - x_point) * (y_point - y1) + 
    #                         q22 * (x_point - x1) * (y_point - y1)) / ((x2 - x1) * (y2 - y1)))
    return elevation_at_point

get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)   #Finds index of max value
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)   #Finds index of min value

def toblers_hiking_function(slope):         
    return 1000*6*math.exp(-3.5*abs(slope+.05))    #Computes speed in m/hr

def objective_f(x):
    way_x1 = x[0]
    way_y1 = x[1]
    way_x2 = x[2]
    way_y2 = x[3]
    topview_distance_segment_1 = math.sqrt((way_x1 - start_point_x)**2 + (way_y1 - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x1 - way_x2)**2 + (way_y1 - way_y2)**2)
    topview_distance_segment_3 = math.sqrt((way_x2 - end_point_x)**2 + (way_y2 - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_way1 = Get_Elevation(x[0],x[1])
    elevation_way2 = Get_Elevation(x[2],x[3])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    true_distance_segment_1 = math.sqrt((elevation_way1 - elevation_start_point)**2 + topview_distance_segment_1**2)
    true_distance_segment_2 = math.sqrt((elevation_way1 - elevation_way2)**2 + topview_distance_segment_2**2)
    true_distance_segment_3 = math.sqrt((elevation_way2 - elevation_end_point)**2 + topview_distance_segment_3**2)

    slope_segment_1 = math.tan((elevation_way1 - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_way2- elevation_way1)/topview_distance_segment_2)
    slope_segment_3 = math.tan((elevation_end_point- elevation_way2)/topview_distance_segment_3)

    speed_segment_1 = toblers_hiking_function(slope_segment_1)
    speed_segment_2 = toblers_hiking_function(slope_segment_2)
    speed_segment_3 = toblers_hiking_function(slope_segment_3)


    time_to_ascend = true_distance_segment_1/speed_segment_1 + true_distance_segment_2/speed_segment_2 + true_distance_segment_3/speed_segment_3

    time_to_ascend = time_to_ascend * 60 # In minutes

    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))
    angle_segment_3 = math.degrees(math.atan(slope_segment_3))

    print('currentX',x)
    # print('Ascenion Time',time_to_ascend)
    print('Angle 1, 2, 3: ', angle_segment_1, angle_segment_2, angle_segment_3)
    # print('Slope 1,2,3', slope_segment_1, slope_segment_2, slope_segment_3 )
    # print('Topview Distance 1,2,3', topview_distance_segment_1, topview_distance_segment_2, topview_distance_segment_3)
    # print('True Distance 1, 2, 3', true_distance_segment_1,true_distance_segment_2, true_distance_segment_3)
    print('Elevation Way Start,1,2,end', elevation_start_point,  elevation_way1, elevation_way2, elevation_end_point)

    return time_to_ascend 

def g1(x):

    way_x1 = x[0]
    way_y1 = x[1]
    way_x2 = x[2]
    way_y2 = x[3]
    topview_distance_segment_1 = math.sqrt((way_x1 - start_point_x)**2 + (way_y1 - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x1 - way_x2)**2 + (way_y1 - way_y2)**2)
    topview_distance_segment_3 = math.sqrt((way_x2 - end_point_x)**2 + (way_y2 - end_point_y)**2)

    largest_segment_difference_minus_bound = max(abs(topview_distance_segment_1 - topview_distance_segment_2), abs(topview_distance_segment_3-topview_distance_segment_2), abs(topview_distance_segment_1 - topview_distance_segment_3)) - max_segment_difference
    print('g1 bound', largest_segment_difference_minus_bound)

    return largest_segment_difference_minus_bound

def g2(x): 
    way_x1 = x[0]
    way_y1 = x[1]
    way_x2 = x[2]
    way_y2 = x[3]
    topview_distance_segment_1 = math.sqrt((way_x1 - start_point_x)**2 + (way_y1 - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x1 - way_x2)**2 + (way_y1 - way_y2)**2)
    topview_distance_segment_3 = math.sqrt((way_x2 - end_point_x)**2 + (way_y2 - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_way1 = Get_Elevation(x[0],x[1])
    elevation_way2 = Get_Elevation(x[2],x[3])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    slope_segment_1 = math.tan((elevation_way1 - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_way2- elevation_way1)/topview_distance_segment_2)
    slope_segment_3 = math.tan((elevation_end_point- elevation_way2)/topview_distance_segment_3)


    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))
    angle_segment_3 = math.degrees(math.atan(slope_segment_3))

    return angle_segment_1 - 15

def g3(x): 
    way_x1 = x[0]
    way_y1 = x[1]
    way_x2 = x[2]
    way_y2 = x[3]
    topview_distance_segment_1 = math.sqrt((way_x1 - start_point_x)**2 + (way_y1 - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x1 - way_x2)**2 + (way_y1 - way_y2)**2)
    topview_distance_segment_3 = math.sqrt((way_x2 - end_point_x)**2 + (way_y2 - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_way1 = Get_Elevation(x[0],x[1])
    elevation_way2 = Get_Elevation(x[2],x[3])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    slope_segment_1 = math.tan((elevation_way1 - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_way2- elevation_way1)/topview_distance_segment_2)
    slope_segment_3 = math.tan((elevation_end_point- elevation_way2)/topview_distance_segment_3)


    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))
    angle_segment_3 = math.degrees(math.atan(slope_segment_3))


    return angle_segment_2 - 15

def g4(x): 
    way_x1 = x[0]
    way_y1 = x[1]
    way_x2 = x[2]
    way_y2 = x[3]
    topview_distance_segment_1 = math.sqrt((way_x1 - start_point_x)**2 + (way_y1 - start_point_y)**2)
    topview_distance_segment_2 = math.sqrt((way_x1 - way_x2)**2 + (way_y1 - way_y2)**2)
    topview_distance_segment_3 = math.sqrt((way_x2 - end_point_x)**2 + (way_y2 - end_point_y)**2)

    elevation_start_point = Get_Elevation(start_point_x,start_point_y)
    elevation_way1 = Get_Elevation(x[0],x[1])
    elevation_way2 = Get_Elevation(x[2],x[3])
    elevation_end_point = Get_Elevation(end_point_x,end_point_y)

    slope_segment_1 = math.tan((elevation_way1 - elevation_start_point)/topview_distance_segment_1)
    slope_segment_2 = math.tan((elevation_way2- elevation_way1)/topview_distance_segment_2)
    slope_segment_3 = math.tan((elevation_end_point- elevation_way2)/topview_distance_segment_3)


    angle_segment_1 = math.degrees(math.atan(slope_segment_1))
    angle_segment_2 = math.degrees(math.atan(slope_segment_2))
    angle_segment_3 = math.degrees(math.atan(slope_segment_3))


    return angle_segment_3 - 15

CONVERT_COORD_TO_METER = 111139

minX = 0
maxX = abs(76.907129 - 76.943693) * CONVERT_COORD_TO_METER
minY = 0
maxY = abs(40.090454 - 40.115401) * CONVERT_COORD_TO_METER

print('Mins and Maxes x/y', minX, maxX, minY, maxY)

x_data, y_data, elevation, transform = Extract_Geotiff_Data()
y_data = y_data - y_data[0]

maxX = x_data[len(x_data) - 1]
maxY = y_data[len(y_data) - 1]

interp_func = RectBivariateSpline(x_data, y_data, elevation)

# print('Mins and Maxes x/y', minX, maxX, minY, maxY)

# print('x_data',x_data)
# print('y_data',y_data)

end_point_idx = get_max_index(elevation)
end_point_y = y_data[end_point_idx[1]]
end_point_x = x_data[end_point_idx[0]]

# print(f"ending index: {end_point_idx}\n")

# print(elevation[end_point_idx])
# print(end_point_x)
# print(end_point_y)

# Manually Chosen Start Point
start_point_x = 500
start_point_y = 500

# print(Get_Elevation(start_point_x,start_point_y))

### START PROBLEM 4_2 ###

### THESE ARE OUR DESIGN VARIABLES ###

spacing_x = (end_point_x - start_point_x)/3
# print('spacing_x',spacing_x)
spacing_y = (end_point_y - start_point_y)/3
# print('spacing_y', spacing_y)
waypoint_x1 = (start_point_x + spacing_x) + 100
waypoint_y1 = (start_point_y + spacing_y) - 100
waypoint_x2 = (end_point_x - spacing_x) - 100
waypoint_y2 = (end_point_y - spacing_y) + 100
### END OF DESING VARIABLES ###

x0 = [waypoint_x1, waypoint_y1,waypoint_x2,waypoint_y2]

print('x0', x0)

optimizer_options = {'disp': True, 'maxiter':500}

geographic_bounds = ((minX,maxX), (minY,maxY),(minX,maxX), (minY,maxY))

max_segment_difference = 600
segment_slope_constraints = [{'type':'ineq', 'fun':g1 },{'type':'ineq', 'fun':g2},{'type':'ineq','fun':g3},{'type':'ineq','fun':g4}]
# segment_slope_constraints = [{'type':'ineq', 'fun':g2},{'type':'ineq','fun':g3},{'type':'ineq','fun':g4}]

res = minimize(objective_f, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
# res = minimize(objective_f, x0, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)
print(res)

print('x*', res.x)

x_star = res.x

# print(objective_f(x0))
# print(elevation[1000][1000])

plt.contour((x_data), (y_data), np.transpose(elevation), 25)
plt.colorbar()

x_points = [start_point_x, x0[0], x0[2], end_point_x]
y_points = [start_point_y, x0[1], x0[3], end_point_y]

x_stars = [start_point_x, x_star[0], x_star[2], end_point_x]
y_stars = [start_point_y, x_star[1], x_star[3], end_point_y]

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