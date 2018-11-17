import unittest

from neuralnet import *


class SoftmaxUnittest(unittest.TestCase):
    def test_softmax_equal_inputs(self):
        inputs = [3, 3, 3, 3]
        layer = SoftmaxLayer()
        outputs = layer.process(inputs)
        self.assertEqual(len(inputs), len(outputs), msg="Softmax should return array of same size as inputs")
        for y in outputs:
            self.assertAlmostEqual(1 / 4, y, places=5,
                                   msg="Softmax on 4 inputs of equal value should return 1/4 for each")

    def test_softmax_single_input(self):
        inputs = [10]
        layer = SoftmaxLayer()
        outputs = layer.process(inputs)
        self.assertEqual(len(inputs), len(outputs), msg="Softmax should return array of same size as inputs")
        self.assertEqual(1, outputs[0], msg="Softmax of length-1 array should be 1")

    def test_softmax_unequal_inputs(self):
        inputs = [3, 4, 1]
        layer = SoftmaxLayer()
        outputs = layer.process(inputs)
        self.assertEqual(len(inputs), len(outputs), msg="Softmax should return array of same size as inputs")
        expected_outputs_total = math.exp(3) + math.exp(4) + math.e
        expected_outputs = [math.exp(3) / expected_outputs_total, math.exp(4) / expected_outputs_total,
                            math.e / expected_outputs_total]
        self.assertEqual(len(inputs), len(outputs), msg="Softmax should return array of same size as inputs")
        for i in range(len(inputs)):
            self.assertAlmostEqual(expected_outputs[i], outputs[i], places=5,
                                   msg="Softmax value differs from expected at index %d" % i)


class AbstractNonOverlapPoolingLayerTest(unittest.TestCase):
    def test_max_of_nums(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = AbstractNonOverlapPoolingLayer((2, 2), numpy.amax)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match")


class NonOverlapMaxpoolLayerTest(unittest.TestCase):
    def test_2x2_tiles(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = NonOverlapMaxpoolLayer((2, 2))
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match")
