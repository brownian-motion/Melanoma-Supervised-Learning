import unittest

from numpy import mean, dtype

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
        outputs = layer.process(inputs).flatten()
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
        expected_outputs = numpy.array([[[6, 8]]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))

    def test_max_of_nums_with_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = AbstractPoolingLayer((2, 2), func=numpy.amax, overlap_tiles=True)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[[6, 7, 8]]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))


class NonOverlapMaxpoolLayerTest(unittest.TestCase):
    def test_2x2_tiles_no_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = MaxpoolLayer((2, 2), overlap_tiles=False)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[[6, 8]]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))

    def test_2x2_tiles_with_overlap(self):
        inputs = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        maxpool_layer = MaxpoolLayer((2, 2), overlap_tiles=True)
        outputs = maxpool_layer.process(inputs)
        expected_outputs = numpy.array([[[6, 7, 8]]])

        self.assertTrue(numpy.array_equal(expected_outputs, outputs),
                        msg="Expected outputs and outputs from maxpool should match (%s) vs (%s)" % (
                            expected_outputs, outputs))


class FullyConnectedLayerTest(unittest.TestCase):
    def test_3x1_layer_with_initial_weights_gives_predicted_result(self):
        layer = FullyConnectedLayer(3, 1, 0.01, 'relu')
        layer.weights = to_row_vector([0.6, 0.7, 0.8])
        layer.bias = to_column_vector([0.9])
        result = layer.process([4, 5, 6])
        expected_result = [.6 * 4 + .7 * 5 + .8 * 6 + 0.9]
        self.assertEqual(len(expected_result), len(result),
                         msg="Expected result and actual result should be the same length")
        for i in range(len(expected_result)):
            self.assertEqual(expected_result[i], result[i],
                             msg="Expected result %.2f differed from actual result %.2f at index %d" % (
                                 expected_result[i], result[i], i))

    def test_2x2_layer_with_handset_weights_gives_predicted_result(self):
        layer = FullyConnectedLayer(2, 2, 0.01, 'relu')
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

    def test_2x2_learns_from_backpropagation(self):
        layer = FullyConnectedLayer(2, 2, 0.001, 'relu')
        layer.weights = numpy.array([[1.0, 2], [4, 5]])  # field is impl-specific
        layer.bias = to_column_vector([3.0, 6])
        sample_input = [10, 11]
        result = layer.process(sample_input, remember_inputs=True)
        for _ in range(3):
            layer.backpropagate(to_column_vector([1, -1]))
            result2 = layer.process(sample_input, remember_inputs=True)
        self.assertLess(result2[0, 0], result[0, 0],
                        msg="Prediction for top of column should have grown smaller, because error indicated it was too large (1 more than true)")
        self.assertGreater(result2[1, 0], result[1, 0],
                           msg="Prediction for bottom of column should have grown smaller, because error indicated it was too small (1 less than true_")

    def test_4x2_learns_from_backpropagation(self):
        layer = FullyConnectedLayer(4, 2, 0.001, 'relu')
        layer.weights = numpy.array([[1.0, 2, 3, 4], [4, 5, 6, 7]])  # field is impl-specific
        layer.bias = to_column_vector([3.0, 6])
        sample_input = [10, 11, 12, 13]
        result = layer.process(sample_input, remember_inputs=True)
        for _ in range(4):
            layer.backpropagate(to_column_vector([1, -1]))
            result2 = layer.process(sample_input, remember_inputs=True)
        self.assertLess(result2[0, 0], result[0, 0],
                        msg="Prediction for top of column should have grown smaller, because error indicated it was too large (1 more than true)")
        self.assertGreater(result2[1, 0], result[1, 0],
                           msg="Prediction for bottom of column should have grown smaller, because error indicated it was too small (1 less than true_")

    def test_4x1_sigmoid_learns_from_backpropagation(self):
        layer = FullyConnectedLayer(4, 1, 0.001, 'sigmoid')
        layer.weights = numpy.array([[.25, .5, .75, -1]])  # field is impl-specific
        layer.bias = to_column_vector([3.0])
        sample_input = [-.5, -1.1, 1.2, 1.3]
        result = layer.process(sample_input, remember_inputs=True)
        for _ in range(4):
            layer.backpropagate(to_column_vector([1]))
            result2 = layer.process(sample_input, remember_inputs=True)
        self.assertLess(result2[0, 0], result[0, 0],
                        msg="Prediction for top of column should have grown smaller, because error indicated it was too large (1 more than true)")


class MeanpoolLayerTest(unittest.TestCase):
    def test_computes_means_correctly_with_2x3_tile(self):
        layer = MeanpoolLayer((2, 3))
        input = numpy.fromiter(range(1, 25), dtype=dtype('f8')).reshape((4, 6))
        results = layer.process(input, remember_inputs=False)
        expected_results = numpy.asarray([[[mean([1, 2, 3, 7, 8, 9]), mean([4, 5, 6, 10, 11, 12])],
                                           [mean([13, 14, 15, 19, 20, 21]), mean([16, 17, 18, 22, 23, 24])]]])
        self.assertEqual(results.shape, expected_results.shape, "Results and expected results should be the same shape")
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    self.assertEqual(results[i, j, k], expected_results[i, j, k],
                                     "Results and expected results differ at spot (%d, %d, %d)" % (i, j, k))

    def test_backpropagation_is_correct_with_2x3_tile(self):
        layer = MeanpoolLayer((2, 3))
        error = numpy.asarray([[1, 2], [-2, -1]])
        expected_error = numpy.asarray(
            [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
             [-2, -2, -2, -1, -1, -1], [-2, -2, -2, -1, -1, -1]]) / 6
        results = layer.backpropagate(error)
        self.assertEqual(results.shape, expected_error.shape,
                         "Calculated error and expected error should be the same shape")
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                self.assertEqual(results[i, j], expected_error[i, j],
                                 "Calculated error and expected error differ at spot (%d, %d)" % (i, j))


class ConvolutionalLayerTest(unittest.TestCase):
    def test_convolutionallayer_with_2d_input_gives_3d_output(self):
        layer = ConvolutionalLayer((2, 2), num_filters=1)
        inputs = numpy.random.rand(4, 5)
        results = layer.process(inputs, remember_inputs=False)
        self.assertSequenceEqual((1, 3, 4), results.shape)

    def test_convolutionallayer_with_3d_input_gives_3d_output(self):
        layer = ConvolutionalLayer((2, 2), num_filters=1)
        inputs = numpy.random.rand(2, 4, 5)
        results = layer.process(inputs, remember_inputs=False)
        self.assertSequenceEqual((2, 3, 4), results.shape)

    def test_convolutional_layer_with_known_weight_gives_predicted_results(self):
        layer = ConvolutionalLayer((2, 2), num_filters=1)
        layer.filters = [numpy.array([[1, 2], [3, 4]])]  # implementation-dependent
        results = layer.process(numpy.array([[1, 1, 5], [2, 2, 2], [1, 1, 1]]))
        expected_results = numpy.array([[[1 * 4 + 1 * 3 + 2 * 2 + 2 * 1, 1 * 4 + 5 * 3 + 2 * 2 + 2 * 1],
                                         [2 * 4 + 2 * 3 + 1 * 2 + 1 * 1, 2 * 4 + 2 * 3 + 1 * 2 + 1 * 1]]])
        self.assertEqual(results.shape, expected_results.shape, "Results and expected results should be the same shape")
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    self.assertEqual(results[i, j, k], expected_results[i, j, k],
                                     "Results and expected results differ at spot (%d, %d, %d)" % (i, j, k))

    def test_convolutional_layer_with_known_weights_and_two_filters_gives_predicted_results(self):
        layer = ConvolutionalLayer((2, 2), num_filters=1)
        layer.filters = [numpy.array([[1, 2], [3, 4]]), numpy.array([[-5, -2], [-3, -4]])]  # implementation-dependent
        results = layer.process(numpy.array([[1, 1, 5], [2, 2, 2], [1, 1, 1]]))
        expected_results = numpy.array([[[1 * 4 + 1 * 3 + 2 * 2 + 2 * 1, 1 * 4 + 5 * 3 + 2 * 2 + 2 * 1],
                                         [2 * 4 + 2 * 3 + 1 * 2 + 1 * 1, 2 * 4 + 2 * 3 + 1 * 2 + 1 * 1]],
                                        [[1 * -4 + 1 * -3 + 2 * -2 + 2 * -5, 1 * -4 + 5 * -3 + 2 * -2 + 2 * -5],
                                         [2 * -4 + 2 * -3 + 1 * -2 + 1 * -5, 2 * -4 + 2 * -3 + 1 * -2 + 1 * -5]]])
        self.assertEqual(results.shape, expected_results.shape, "Results and expected results should be the same shape")
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    self.assertEqual(results[i, j, k], expected_results[i, j, k],
                                     "Results and expected results differ at spot (%d, %d, %d)" % (i, j, k))

    def test_convolutional_layer_learns(self):
        layer = ConvolutionalLayer((2, 2), num_filters=1)
        sample_input = numpy.array([[[1.0, 1, 5], [2, 0, 2], [1, 1, 1]]])
        results = layer.process(sample_input, remember_inputs=True)
        for _ in range(4):
            layer.backpropagate(numpy.array([[[1.0, 0], [0, -1]]]))
            results2 = layer.process(sample_input, remember_inputs=True)
        self.assertLess(results2[0, 0, 0], results[0, 0, 0],
                        msg="Prediction for top left should have grown smaller, because error indicated it was too large (1 more than true value)")
        self.assertGreater(results2[0, 1, 1], results[0, 1, 1],
                           msg="Prediction for bottom right should have grown greater, because error indicated it was too small (1 less than true value)")


class LinearNeuralNetworkTest(unittest.TestCase):
    def test_linear_neural_net_functions_at_all(self):
        net = SimpleNeuralBinaryClassifier()
        net.add_layer(ConvolutionalLayer((2, 2)))
        net.add_layer(FullyConnectedLayer(4, 2))
        net.add_layer(SoftmaxLayer())

        # ignore the results, just run the _process to see what comes out
        results = net._process(numpy.array([[1.0, 1, 5], [2, 2, 2], [1, 1, 1]]))
        self.assertEqual(2, results.size, msg="Number of outputs doesn't match what's expected")
        self.assertTrue(is_column_vector(results))

    def test_linear_neural_net_learns(self):
        net = SimpleNeuralBinaryClassifier()
        net.add_layer(ConvolutionalLayer((2, 2)))
        net.add_layer(FullyConnectedLayer(4, 2))
        net.add_layer(SoftmaxLayer())

        # ignore the results, just run the _process to see what comes out
        sample_inputs = numpy.array([[1.0, 1, 5], [2, 2, 2], [1, 1, 1]])
        results = net._process(sample_inputs)
        self.assertEqual(2, results.size, msg="Number of outputs doesn't match what's expected")
        self.assertTrue(is_column_vector(results))
        net._backpropagate(numpy.add(to_column_vector([-1, 1]), results))
        results2 = net._process(sample_inputs)
        self.assertGreater(results2[0, 0], results[0, 0],
                           msg="Top output should have grown larger, because error indicated that it was too small (1 less than true value)")
        self.assertLess(results2[1, 0], results[1, 0],
                        msg="Bottom output should have grown smaller, because error indicated that it was too large (1 more than true value)")
