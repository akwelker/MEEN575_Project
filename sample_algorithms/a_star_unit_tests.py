# Adam Welker       MEEN 575        Winter 23
#
# a_star_unit_tests.py -- a file to test my A* implementation


from a_star import *
import unittest as test



class Test_Node_Path(test.TestCase):

    def test_root_path(self):

        root = Node(parent=None,location=(0,0))

        self.assertEqual(root.getPath(), [(0,0)])

    def test_leaf_path(self):

        root = Node(parent=None,location=(0,0))
        curr_node = root

        for i in range(1, 11):
            
            prev_node = curr_node # save the parent node

            curr_node = Node(parent=prev_node, location=(i,i), g=1,h=1)

        test_path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

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






if __name__ == '__main__':

    test.main()