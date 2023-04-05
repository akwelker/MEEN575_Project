# Adam Welker   MEEN 575      Winter 23
#
# a file that implements the eggshell function

import numpy as np

def egg_shell(x):

    try:

        x1 = x[0]
        x2 = x[1]

    except:

        raise Exception("x does not have two elements!")
    

    f = 0.1*x1**2 + 0.1*x2**2 - np.cos(3*x1) - np.cos(3*x2)

    return f