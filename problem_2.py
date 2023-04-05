# Problem_2.py -- Based on the file General_Project_Setup.py file
# created by my team make Isaac. I've adapted the file to use the
# Binary Genetic Algorithm that I created for problem 1
#
# - Adam Welker


import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import minimize, NonlinearConstraint, Bounds 
from binary_GA import *


from mountain_geography import * # helper functions to do with geography
from path_problem import * # Defines path problem and constraints

# Just some setup stuff
range_noise = 150

for i in range(num_waypoints):
    x0[2*i] = (start_point_x + (i+1)*spacing_x) + random.uniform(-1*range_noise, range_noise)
    x0[2*i + 1] = (start_point_y + (i+1)*spacing_y) + random.uniform(-1*range_noise, range_noise)

# ========================= HERE IS WHERE ADAM'S STUFF BEGINS ===================================

def penalty_obj_function(x):

    penalty = 0.005

    sumation = 0

    sumation += objective_f(x)

    constrnt = g1(x)

    for val in constrnt:

        sumation += penalty * max([0, val])**2

    return sumation

ga_optimizer = Binary_GA(500,2500,32,verbose=True) # Initalize the optimizer

result = ga_optimizer.GA_optimization(penalty_obj_function,5000,200, num_waypoints*2)

points = result[0]
time = objective_f(result[0])

print(g1(points))

x_points = [start_point_x]
y_points = [start_point_y]

for i in range(0, len(points)):

    if i % 2 == 0:

        x_points.append(points[i])

    else:

        y_points.append(points[i])

x_points.append(end_point_x)
y_points.append(end_point_y)

# ========================= HERE IS WHERE ADAM'S STUFF ENDS =====================================

plt.contour((x_data), (y_data), np.transpose(elevation), 25)
plt.colorbar()

plt.plot(x_points,y_points,'r*-')

# Add axis labels and a title
plt.xlabel('X Vals (meters)')
plt.ylabel('Y Vals (meters)')
plt.title(f'GA Generated Hiking Path\n Estimated Traversal Time: {round(time)} minutes')
plt.legend(['Optimally Found Path'])

# Show the plot
plt.show()