import unittest

from decisiontree.DecisionTree import *

class Test(unittest.TestCase):
    def test_grow_tree_doesnt_crash(self):
        inputs = np.array([[1,2,3,4,5,0],[6,7,8,9,10,1]])
        tree = growTree(inputs)



        print("Test: Grow Tree doesn't crash")

    def test_predict(self):
        inputs = np.array([[1,2,3,4,5,0],[6,7,8,9,10,1]])
        tree = growTree(inputs)

        results = tree.predict(inputs)

        print("Test: Predict doesn't crash")
        self.assertEqual(0,results[0])
        self.assertEqual(1,results[1])



