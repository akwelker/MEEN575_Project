# Adam Welker       MEEN 575        Winter 23
#
# a_star.py -- Uses a_star to plot a path up a mountain

import numpy as np
from queue import PriorityQueue

class PathFinder():
    """
    Class capable of returning the optimal path through g and h cost maps.
    Assumes 2 degrees of freedom

    @var start: a tuple of 2 ints (>= 0) that denotes the starting position
    @var end: a tuple of 2 ints (>= 0) that denotes the goal position
    @var g: a (n x m) sized numpy array containing local cost of traversal g
    @var h: a (n x m) sized numpy array containing local cost of hueristic h
    """

    _start = None
    _end = None
    _g = None
    _h = None

    _queue = PriorityQueue()

    def __init__(self, start, end, g, h) -> None:

         # Ensure g and h are valid size
        try: 
            assert len(g.size) == 2
            assert len(h.size) == 2
            assert g.size == h.size

            self._g = g
            
        except:

            raise Exception('ERROR: g and h are not equally sized (n x m) numpy arrays')
        
        # Ensure that start and end postions are valid
        try:
            assert len(start) == 2
            assert start[0] >= 0 and start[0] < self._g.size[0]
            assert start[1] >= 0 and start[1] < self._g.size[1]

            self._start = start

        except:

            raise Exception('ERROR: start position is not a valid tuple')

        try:
            assert len(start) == 2
            assert end[0] >= 0 and end[0] < self._g.size[0]
            assert end[1] >= 0 and end[1] < self._g.size[1]

            self._end = end

        except:
            
            raise Exception('ERROR: end position is not a valid tuple')
            



    def a_star(max_iter = np.inf):
        """
        Returns the optimal path through g and h cost maps.
        Assumes 2 degrees of freedom.
        """
        
        # Ensure g and h are valid
        try: 
            assert len(g.size) == 2
            assert len(h.size) == 2
            assert g.size == h.size

        except:

            raise Exception('ERROR: g and h are not equally sized (n x m) numpy arrays')







class Node:
    """
    A Class that represents one node (and correlating path) visited by the A* path planning node
    in our topography
    """

    _parent = None # the node we came from
    _location = (None, None) # the grid location of the node
    _cost = 0 # the total cost of the path leading up to the node (i.e.: g + H + f(n-1))

    def __init__(self, parent = None, location = None, g = 0, h = 0) -> None:
        
        self._parent = parent
        self._location = location
        self._cost = g + h

        # if there's a parent, add the parent's cost
        if self._parent != None:

            self._cost += self._parent._cost

    
    def getPath(self):
        """
        Function that returns the path taken to the current node
        """

        path = []

        # If there's a parent path, get it recursively
        if self._parent != None:

            path = self._parent.getPath()

        # now add where we are currently

        path.append(self._location)

        # return the path
        return path
    
    
    # Accessors and such ###############################
    def getCost(self):
        """
        Returns the total path cost of the node that is being examined
        """ 
        return self._cost


    



    

