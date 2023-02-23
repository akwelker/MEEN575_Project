# Adam Welker       MEEN 575        Winter 23
#
# generate_sample_mountain.py -- A program that 
# generates the typography, cost and passablility map
# given the a 2-D Guassian Distribution


import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


#Define parameters of function

mean_x = 0
mean_y = 0

covariance = 0
sigma_x = 50
sigma_y = 50

SIZE_X = 25
SIZE_Y = 25

# Now define the terrain function
x, y = np.mgrid[-SIZE_X:SIZE_X:1, -SIZE_X:SIZE_X:1]



pos = np.dstack((x,y))

PDF = stats.multivariate_normal([mean_x, mean_y], [[sigma_x, covariance],[covariance, sigma_y]])

terrain = PDF.pdf(pos)

x_set = np.linspace(-SIZE_X,SIZE_X,terrain.shape[0])
y_set = np.linspace(-SIZE_Y,SIZE_Y,terrain.shape[1])

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(x,y,terrain, alpha=0.75)
# Now Specify the starting and ending points

start = (0,49)

end = (int(terrain.shape[0]/2), int(terrain.shape[1]/2))

ax.scatter(x_set[start[0]],y_set[start[1]],terrain[start[0], start[1]], color="r")
ax.scatter(x_set[end[0]],y_set[end[1]],terrain[end[0], end[1]], marker="^", color="r")


# Now add a cost matrix that denotes impassible zones
traversal_cost = np.zeros(terrain.shape)

#We'll pretend that the southeast side of the 
# map is private property and cannot be passed

for i in range(int(terrain.shape[0]/2), terrain.shape[0]):
    for j in range(int(terrain.shape[1]/2), terrain.shape[1]):

        traversal_cost[i,j] = np.inf

print(traversal_cost)


if __name__ == '__main__':
    plt.show()
