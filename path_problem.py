import numpy as np
import random
import math
from mountain_geography import *
from scipy.optimize import Bounds, NonlinearConstraint



def Compute_Linear_Path_Noisy(num_waypoints, start_point_x, start_point_y, end_point_x, end_point_y, range_noise):
    x0 = np.zeros(num_waypoints*2)
    spacing_x = (end_point_x - start_point_x)/(num_waypoints+1)
    spacing_y = (end_point_y - start_point_y)/(num_waypoints+1)

    x_array = np.zeros(num_waypoints)
    y_array = np.zeros(num_waypoints)

    for i in range(num_waypoints):
        x0[2*i] = (start_point_x + (i+1)*spacing_x) + random.uniform(-1*range_noise, range_noise)
        x_array[i] = x0[2*i]
        x0[2*i + 1] = (start_point_y + (i+1)*spacing_y) + random.uniform(-1*range_noise, range_noise)
        y_array[i] = x0[2*i + 1]

    return x0, x_array, y_array

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

# Define the bounds for each variable, even numbered indices (x_points) get xBounds, odd numbered indices (y_points) get yBounds
lower_bounds = [minX if i % 2 == 0 else minY for i in range(x0.size)]
upper_bounds = [maxX if i % 2 == 0 else maxY for i in range(x0.size)]

geographic_bounds = Bounds(lower_bounds, upper_bounds)

slope_constraint_degrees = 12
# segment_slope_constraints = [{'type':'ineq', 'fun':g1 }]
segment_slope_constraints = NonlinearConstraint(g1, lb = -np.inf, ub = 0, jac = jac_constraint)

