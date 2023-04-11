# Adam Welker   MEEN 575  Winter 23
#
# hw_4_demo.py -- given a guassian distribution "moutain",
# the demo proves that a_star can find a way up

import numpy as np
from matplotlib import pyplot as plt
import a_star

import mountain_geography
import path_problem

# ========================== Extract Data and Create H hueristic Matrix =======================
x_points, y_points, ell_data = mountain_geography.Extract_Geotiff_Raw()

#Create max and min index functions
get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)

trail_end = get_max_index(ell_data)

def get_location(index):

    i = index[0]
    j = index[1]

    loc_x = x_points[i][j]
    loc_y = y_points[i][j]
    loc_z = ell_data[i][j]

    return np.array([loc_x, loc_y, loc_z])


trail_end_loc = get_location(trail_end).T

print(f'End of Trail @ {(trail_end_loc.item(0), trail_end_loc.item(1), trail_end_loc.item(2))}')

trail_head = (0,0)

# Make a map of all point data as well as h hueristic matrix
master_map = []
h = np.zeros_like(x_points)

print("MAKING H MATRIX!")
for i in range(0, len(x_points)):

    new_row = []

    for j in range(0, len(x_points[0])):
        
        # add location tuple to master map
        new_row.append([x_points[i][j], y_points[i][j], ell_data[i][j]])

        # Define hueristic as distance to trail end
        loc_x = x_points[i][j]
        loc_y = y_points[i][j]
        loc_z = ell_data[i][j]

        h[i][j] = np.linalg.norm(get_location((i,j)).T - trail_end_loc)



    master_map.append(new_row)

print("H MATRIX CREATED!")

# ==================================================== MAKE G COST FUNCTION ==================================

# Define slope constraint
def slope_constrnt(dx, dz) -> float:

    MAX_SLOPE_ANGLE = np.radians(20.0)

    angle = np.arctan2(dz,dx)

    return angle - MAX_SLOPE_ANGLE


# Define the cost function g 
def g_tobler(location, parent: a_star.Node):

    CNSTRNT_GAIN = 0.05

    # Get beginning and ending points
    destination = get_location(location)
    
    beginning = get_location(parent.getLocation())

    # Implement Tobler's Hiking Function
    dx = destination[0] - beginning[0]
    dy = destination[1] - beginning[1]
    dz = destination[2] - beginning[2]

    flat_distance = np.sqrt(dx**2 + dy**2)
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    walking_speed = 6*np.exp(-3.5 * np.abs(dz/flat_distance + 0.05))

    time = walking_speed/distance

    cnstrnt_cost = CNSTRNT_GAIN * slope_constrnt(flat_distance, dz)

    return time + cnstrnt_cost

#====================================== call A* path plan =================================================

path_planner = a_star.A_Star((0,0), trail_end, np.zeros_like(h), h)

#path_planner.solve(func_g= g_tobler, verbose = True)

# Then plot the given path

# plt.contour(x_points, y_points, ell_data, 50)
# plt.show()