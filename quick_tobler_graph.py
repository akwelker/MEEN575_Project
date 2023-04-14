# Adam Welker
#
# This program makes a demo graph of the tobler fuction


from matplotlib import pyplot as plt
import numpy as np


domain = np.linspace(-90,90,1000)

def tobler_function(theta):

    dh_dx = np.tan(np.radians(theta))

    return 6.0*np.exp(-3.5*np.abs(dh_dx + 0.05))


range = tobler_function(domain)

plt.plot(domain,range)
plt.xlabel("Trail Slope (Deg.)")
plt.ylabel("Walking Speed (km/h)")
plt.title('Tobler Function')
plt.show()