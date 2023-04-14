# Adam Welker   MEEN 575  Winter 23
#
# hw_4_demo.py -- given a guassian distribution "moutain",
# the demo proves that a_star can find a way up

import numpy as np
from matplotlib import pyplot as plt
import a_star

import mountain_geography

# ========================== Extract Data and Create H hueristic Matrix =======================
x_points, y_points, ell_data = mountain_geography.Extract_Geotiff_Raw()

#Create max and min index functions
get_max_index = lambda x: np.unravel_index(np.argmax(x), x.shape)
get_min_index = lambda x: np.unravel_index(np.argmin(x), x.shape)

trail_end = (x_points.shape[0]- 30, 37)

def get_location(index):

    i = index[0]
    j = index[1]

    loc_x = x_points[i][j]
    loc_y = y_points[i][j]
    loc_z = ell_data[i][j]

    return np.array([loc_x, loc_y, loc_z])


trail_end_loc = get_location(trail_end).T

print(f'End of Trail @ {(trail_end_loc.item(0), trail_end_loc.item(1), trail_end_loc.item(2))}')

trail_head = (x_points.shape[1] - 4, 3)
print(trail_head)

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

        h[i][j] = np.linalg.norm(get_location((i,j)).T - trail_end_loc) ** 6.0
        # h[i][j] = np.linalg.norm(np.array([i-trail_end[0], j-trail_end[1]]))

    master_map.append(new_row)

print("H MATRIX CREATED!")

print(x_points.shape)

# ==================================================== MAKE G COST FUNCTION ==================================

# Define slope constraint
def slope_constrnt(dx, dz) -> float:

    MAX_SLOPE_ANGLE = np.radians(45.0)

    angle = np.arctan2(dz,dx)

    return abs(angle - MAX_SLOPE_ANGLE)



# Define the cost function g 
def g_tobler(location, parent: a_star.Node):

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

    time = distance/walking_speed

    cnstrnt_cost = slope_constrnt(flat_distance, dz)

    return time + cnstrnt_cost

#====================================== call A* path plan =================================================
path_planner = a_star.A_Star(trail_head, trail_end, np.zeros_like(h), h)
path, cost = path_planner.solve(func_g= g_tobler, verbose = True, max_iter=1e5)

# These are sample values
# path = [[42, 3], [41, 4], [40, 5], [39, 6], [38, 7], [37, 8], [36, 9], [35, 10], [34, 11], [33, 12], [32, 13], [31, 14], [30, 15], [29, 16], [29, 17], [29, 18], [29, 19], [29, 20], [29, 21], [29, 22], [29, 23], [29, 24], [29, 25], [29, 26], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31], [29, 32], [29, 33], [29, 34], [29, 35], [29, 36], [29, 37]]
# itrs =  1283960
print(path)



time_final = 0

for i in range(0, len(path)-1):

    beginning = get_location(path[i])
    destination = get_location(path[i+1])

    # Implement Tobler's Hiking Function
    dx = destination[0] - beginning[0]
    dy = destination[1] - beginning[1]
    dz = destination[2] - beginning[2]

    flat_distance = np.sqrt(dx**2 + dy**2)
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    walking_speed = 6*np.exp(-3.5 * np.abs(dz/flat_distance + 0.05)) * 0.277778 # Use kph conversion factor

    time = distance/walking_speed

    time_final += time


# Then plot the given path

path_x = []
path_y = []

for point in path:

    path_x.append(master_map[point[0]][point[1]][0])
    path_y.append(master_map[point[0]][point[1]][1])

plt.contour(x_points, y_points, ell_data, 50)
plt.plot(path_x,path_y, 'r-')

plt.plot(get_location(start)[0], get_location(start)[1], 'xr')
plt.plot(get_location(trail_end)[0], get_location(trail_end)[1], 'xr')
plt.title("Optimal Path Found Using Modified A*\n" + f"Traversal Time: {time_final/60.} min.")
plt.show()