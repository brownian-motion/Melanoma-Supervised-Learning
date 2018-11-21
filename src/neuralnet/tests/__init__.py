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


class AbstractPoolingLayerTest(unittest.TestCase):
    def test_max_of_nums_no_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = AbstractPoolingLayer((2, 2), func=numpy.amax, overlap_tiles=False)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))

    def test_max_of_nums_with_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = AbstractPoolingLayer((2, 2), func=numpy.amax, overlap_tiles=True)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 7, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))


class NonOverlapMaxpoolLayerTest(unittest.TestCase):
    def test_2x2_tiles_no_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = MaxpoolLayer((2, 2), overlap_tiles=False)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))

    def test_2x2_tiles_with_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = MaxpoolLayer((2, 2), overlap_tiles=True)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[6, 7, 8]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))


class FullyConnectedLayerTest(unittest.TestCase):
    def test_3x1_layer_with_initial_weights_gives_predicted_result(self):
        layer = FullyConnectedLayer(0.01, 3, 1, 'relu')
        layer.weights = numpy.array([[.6], [.7], [.8]])
        result = layer.process([4, 5, 6])
        expected_result = [4]
        self.assertEqual(len(expected_result), len(result),
                         msg="Expected result and actual result should be the same length")
        for i in range(len(expected_result)):
            self.assertEqual(expected_result[i], result[i],
                             msg="Expected result %.2f differed from actual result %.2f at index %d" % (
                                 expected_result[i], result[i], i))

    def test_2x2_layer_with_handset_weights_gives_predicted_result(self):
        layer = FullyConnectedLayer(0.01, 2, 2, 'relu')
        layer.weights = numpy.array([[1, 2], [4, 5]])  # field is impl-specific
        layer.bias = to_column_vector([3, 6])
        result = layer.process([10, 11])
        expected_result = [10 + 22 + 3, 40 + 55 + 6]
        self.assertEqual(len(expected_result), len(result),
                         msg="Expected result and actual result should be the same length")
        for i in range(len(expected_result)):
            self.assertEqual(expected_result[i], result[i],
                             msg="Expected result %.2f differed from actual result %.2f at index %d" % (
                                 expected_result[i], result[i], i))
