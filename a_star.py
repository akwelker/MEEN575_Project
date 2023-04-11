# Adam Welker       MEEN 575        Winter 23
#
# a_star.py -- Uses a_star to plot a path up a mountain

import numpy as np
import heapq

class A_Star():
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

    _queue = []

    def __init__(self, start, end, g, h) -> None:

         # Ensure g and h are valid size
        try: 
            assert len(g.shape) == 2
            assert len(h.shape) == 2
            assert g.shape == h.shape

            self._g = g
            
        except:

            raise Exception('ERROR: g and h are not equally sized (n x m) numpy arrays')
        
        # Ensure that start and end postions are valid
        try:
            assert len(start) == 2
            assert start[0] >= 0 and start[0] < self._g.shape[0]
            assert start[1] >= 0 and start[1] < self._g.shape[1]

            self._start = start

        except:

            raise Exception('ERROR: start position is not a valid tuple')

        try:
            assert len(start) == 2
            assert end[0] >= 0 and end[0] < self._g.shape[0]
            assert end[1] >= 0 and end[1] < self._g.shape[1]

            self._end = end

        except:
            
            raise Exception('ERROR: end position is not a valid tuple')
        

        self._start = start
        self._end = end
        self._g = g
        self._h = h
        self._queue = []



    def solve(self, max_iter = np.inf, func_g = None, func_h = None, verbose = False):
        """
        Returns the optimal path through g and h cost maps.
        Assumes 2 degrees of freedom.
        """

        if verbose:

            print(bcolors.OKGREEN + '==== A* Path Planning Started ====' + bcolors.ENDC)
            print(bcolors.OKCYAN + "Max Iterations: " + bcolors.ENDC + f'{max_iter}')

        # Reset queue and make the root node
        self._queue = []
        root = Node(location = self._start)
        heapq.heappush(self._queue, root)

        
        # Now make result variables -- the current node and if we've reached the goal
        curr_node = None
        reached_end = False

        # Now begin the search

        itr = 0

        while itr < max_iter and reached_end == False and len(self._queue) > 0:

            curr_node = heapq.heappop(self._queue)

            # Check if we've reached our goal

            if curr_node.getLocation() == self._end:

                reached_end = True

            else:

                self._explore(curr_node, func_g=func_g, func_h=func_h)
                itr += 1

        
        if not reached_end:
            
            print(bcolors.WARNING + "WARNING:" + bcolors.ENDC +" A* did not reach its goal!")
            
            return [], np.inf
        
        if verbose:

            print(bcolors.OKGREEN + "==== Path Found! ====" + bcolors.ENDC)

        return curr_node.getPath(), curr_node.getCost()

            
    






    # Given a node, will explore around its exterior
    def _explore(self, node, func_g = None, func_h = None):

        location = node._location

        #explore n
        if location[0] > 0:
            
            new_location = np.copy(location)
            new_location[0] -= 1
            self._search_node(node, new_location, func_g=func_g, func_h=func_h)

        # explore s
        if location[0] < self._g.shape[0] - 1:

            new_location = np.copy(location)
            new_location[0] += 1
            self._search_node(node, new_location, func_g=func_g, func_h=func_h)


        # explore e
        if location[1] > 0:
            
            new_location = np.copy(location)
            new_location[1] -= 1
            self._search_node(node, new_location, func_g=func_g, func_h=func_h)
        # explore w
        if location[1] < self._g.shape[1] - 1:

            new_location = np.copy(location)
            new_location[1] += 1
            self._search_node(node, new_location, func_g=func_g, func_h=func_h)


    # location and a parent, will add a node to the search queue
    def _search_node(self, parent, location, func_g = None, func_h = None):
        """
        Given a parent node, and a location,
        will add a new node to the prioirty queue
        """

        #if we're passed a function, use that to calculate hueristic and cost
        # else use the localized cost
        if func_g == None:
            g_cost = self._g[location[0]][location[1]]
        else:
            g_cost = func_g((location[0], location[1]), parent)
        
        if func_h == None:
            h_cost = self._h[location[0]][location[1]]
        else:
            h_cost = func_h((location[0], location[1]), parent)

        new_node = Node(parent, location, g=g_cost, h= h_cost)

        heapq.heappush(self._queue, new_node)

        



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

        path.append(self.getLocation())

        # return the path
        return path
    
    
    # Accessors and such ###############################
    def getCost(self):
        """
        Returns the total path cost of the node that is being examined
        """ 
        return self._cost
    
    def getLocation(self):
        """
        Return the location as a python list
        """
        loc_x = self._location[0]
        loc_y = self._location[1]

        return [loc_x, loc_y]
    
    # To String Method

    def __str__(self) -> str:
        
        loc_x = self._location[0]
        loc_y = self._location[1]

        return f'Node(Cost: {self.getCost()}, Location: {loc_x}, {loc_y})'

    

    # Comparitor methods

    def __eq__ (self, other):

        if isinstance(other, Node):

            return self.getCost() == other.getCost()
        
        elif other == None:

                return False    
        else:

            raise Exception("Nodes Must Be Compared to Other Nodes")

    def __gt__(self, other):

        if isinstance(other, Node):

            return self.getCost() > other.getCost()
        
        elif other == None:

                return False    
        else:

           raise Exception("Nodes Must Be Compared to Other Nodes")

    def __lt__(self, other):

        if isinstance(other, Node):

            return self.getCost() < other.getCost()
        
        elif other == None:

                return False    
        
        else:

           raise Exception("Nodes Must Be Compared to Other Nodes")
    

# NGL got this from stack overflow
# --> https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



    

