# Adam Welker       MEEN 575        Winter 23
#
# a_star_unit_tests.py -- a file to test my A* implementation


from a_star import *
import unittest as test
from a_star import *
import numpy as np


# Unit tests for node class
class Test_Node(test.TestCase):

    def test_root_path(self):

        root = Node(parent=None,location=(0,0))

        self.assertEqual(root.getPath(), [[0,0]])

    def test_leaf_path(self):

        root = Node(parent=None,location=(0,0))
        curr_node = root

        for i in range(1, 11):
            
            prev_node = curr_node # save the parent node

            curr_node = Node(parent=prev_node, location=(i,i), g=1,h=1)

        test_path = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]

        self.assertEqual(test_path, curr_node.getPath())


class Test_Node_Cost(test.TestCase):

    def test_root_cost(self):

        root = Node(parent=None,location=(0,0))

        self.assertEqual(root.getCost(), 0)

        root = Node(parent=None,location=(0,0), g=1, h=2)

        self.assertEqual(root.getCost(), 3)


    def test_leaf_cost(self):

        root = Node(parent=None,location=(0,0))
        curr_node = root

        for i in range(1, 11):
            
            prev_node = curr_node # save the parent node

            curr_node = Node(parent=prev_node, location=(i,i), g=1,h=1)

        self.assertEqual(curr_node.getCost(), 20)


class Test_A_Star(test.TestCase):

    def test_expore(self):

        g = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        
        h = np.zeros_like(g)

        searcher = A_Star([0,0], [1,1], g, h)

        root = Node(location=[1,1])

        searcher._explore(root)  

        self.assertEqual(len(searcher._queue), 4) 

        # Test corner
        searcher = A_Star([0,0], [1,1], g, h)

        root = Node(location=[0,0])

        searcher._explore(root)  

        self.assertEqual(len(searcher._queue), 2) 

        # Test another corner
        searcher = A_Star([0,0], [1,1], g, h)

        root = Node(location=[2,2])

        searcher._explore(root)  

        self.assertEqual(len(searcher._queue), 2) 


    def test_astar(self):

        g = np.array([[1,2,3],
                      [4,np.inf,6],
                      [7,8,9]])
        
        h = np.array([[4,3,2],
                      [3,2,1],
                      [2,1,0]])

        searcher = A_Star([0,0], [2,2], g, h)

        path, cost = searcher.solve()
        
        self.assertEqual(path, [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])
        self.assertEqual(cost, 26)

        g = np.array([[1,2,3],
                      [4,-5,6],
                      [7,8,9]])
        
        h = np.array([[4,3,2],
                      [3,2,1],
                      [2,1,0]])

        searcher = A_Star([0,0], [2,2], g, h)

        path, cost = searcher.solve()

        print(path)
        print(cost)
    
    
        self.assertEqual(path, [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]])
        self.assertEqual(cost, 18)


        g = np.array([[1,2,3],
                      [4,-5,6],
                      [7,8,np.inf]])
        
        h = np.array([[4,3,2],
                      [3,2,1],
                      [2,1,0]])

        searcher = A_Star([0,0], [2,2], g, h)

        path, cost = searcher.solve(max_iter= 10000)
        
        self.assertEqual(path, [])
        self.assertEqual(cost, np.inf)






if __name__ == '__main__':

    test.main()