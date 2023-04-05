import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import minimize, NonlinearConstraint, Bounds 


from mountain_geography import * # helper functions to do with geography
from path_problem import * # Defines path problem and constraints

# print('Mins and Maxes x/y', minX, maxX, minY, maxY)

range_noise = 150

for i in range(num_waypoints):
    x0[2*i] = (start_point_x + (i+1)*spacing_x) + random.uniform(-1*range_noise, range_noise)
    x0[2*i + 1] = (start_point_y + (i+1)*spacing_y) + random.uniform(-1*range_noise, range_noise)

# print('x0', x0)
print('initial time', objective_f(x0))

optimizer_options = {'disp': False, 'maxiter':1000}

# res = minimize(objective_f, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options)

# res = minimize(obj, x0, bounds=geographic_bounds, tol=1e-7, options=optimizer_options, jac = True)


res = minimize(objective_f, x0, bounds=geographic_bounds, method = 'Powell' , tol=1e-7, options=optimizer_options)

# res = minimize(obj, x0, constraints=segment_slope_constraints, bounds=geographic_bounds, tol=1e-7, options=optimizer_options, jac = True)
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