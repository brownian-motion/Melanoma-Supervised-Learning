import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import convolve

from cross_entropy import binary_cross_entropy
from neuralnet.activation import *
from neuralnet.vector_utils import *


class ActivationLayer:
    '''
    Implements a softmax neural network layer.
    This normalizes input values to the range [0,1],
    such that the sum of the outputs add to 1.
    '''

    def __init__(self, activation_function, activation_deriv):
        self.activation_function = activation_function
        self.activation_deriv = activation_deriv
        self._last_inputs = []
        self._last_outputs = []

    def process(self, inputs, remember_inputs=False):
        '''
        Performs an activation on the given inputs.
        Given an array of N input values,
        normalizes those inputs to the range [0,1]
        such that they sum to 1,
        and returns those N normalized outputs.
        :param inputs: an array (or iterable) of size N
        :return: the N inputs, normalized using softmax
        '''
        if not is_column_vector(inputs):
            inputs = to_column_vector(inputs)
        assert is_column_vector(inputs)

        results = to_column_vector(self.activation_function(inputs))
        assert is_column_vector(results)
        assert results.size == inputs.size

        if remember_inputs:
            self._last_inputs.append(inputs)
            self._last_outputs.append(results)

        return results

    def backpropagate(self, true_ys):
        '''
        Updates any relevant weights (none, for this simple activation layer)
        and then returns the error for the previous layer.
        :param true_ys: the error in what this layer output
        :return: the error for the previous layer, calculated using the true observations,
        the remembered last inputs and outputs through this neuron,
        and the derivative of the activation function.
        '''
        last_outputs = self._last_outputs.pop()
        last_inputs = self._last_inputs.pop()
        return (true_ys - last_outputs) * self.activation_deriv(last_inputs)


class SoftmaxLayer(ActivationLayer):
    def __init__(self):
        super().__init__(vec_softmax, vec_softmax_deriv)

    def backpropagate(self, true_ys):
        '''
        Updates any relevant weights (none, for this simple activation layer)
        and then returns the error for the previous layer.
        :param true_ys: the error in what this layer output
        :return: the error for the previous layer
        '''
        last_outputs = self._last_outputs.pop()
        return true_ys - last_outputs


class SigmoidLayer(ActivationLayer):
    def __init__(self):
        super().__init__(vec_sigmoid, vec_sigmoid_deriv)


class AbstractConvolutionalLayer:
    '''
    Implements basic convolution for a neural net layer,
    which breaks up an input matrix into tiles of a particular shape
    so that each tile can be processed separately.
    '''

    def __init__(self, tile_shape, overlap_tiles=False):
        if len(tile_shape) != 2:
            raise ValueError(
                "This pooling layer implementation only knows how to handle 2-D tiles, inputs, and outputs")
        self.tile_shape = tile_shape
        self._overlap_tiles = overlap_tiles
        self._last_inputs = []
        self._last_outputs = []

    def _get_tile(self, tile_coord, input, tile_shape):
        if len(tile_shape) != 2 or len(tile_coord) != 2:
            raise ValueError("Sorry, this convolutional layer only knows how to get 2-D tiles from input matrices")

        if self._overlap_tiles:
            left = tile_coord[0]
        else:
            left = tile_coord[0] * tile_shape[0]

        right = left + tile_shape[0]

        if self._overlap_tiles:
            top = tile_coord[1]
        else:
            top = tile_coord[1] * tile_shape[1]

        bottom = top + tile_shape[1]

        tile = input[left:right, top:bottom]
        assert tile.shape == tile_shape
        return tile

    def convolve2d(self, input_mat, convolution_func):
        assert len(input_mat.shape) == 2
        output_shape = self._compute_output_shape(input_mat.shape, self.tile_shape)
        output = numpy.zeros(output_shape)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                tile = self._get_tile((i, j), input_mat, self.tile_shape)
                output[i, j] = convolution_func(tile)
        return output

    def _compute_output_shape(self, input_shape, tile_shape):
        if self._overlap_tiles:
            return self._compute_output_shape_with_overlap(input_shape, tile_shape)
        return self._compute_output_shape_non_overlap(input_shape, tile_shape)

    @staticmethod
    def _compute_output_shape_non_overlap(input_shape, tile_shape):
        return tuple(math.ceil(input_shape[i] / tile_shape[i]) for i in range(len(tile_shape)))

    @staticmethod
    def _compute_output_shape_with_overlap(input_shape, tile_shape):
        return tuple(math.ceil(input_shape[i] - tile_shape[i] + 1) for i in range(len(tile_shape)))

    def _compute_input_shape(self, output_shape, tile_shape):
        if self._overlap_tiles:
            return self._compute_input_shape_with_overlap(output_shape, tile_shape)
        return self._compute_input_shape_non_overlap(output_shape, tile_shape)

    @staticmethod
    def _compute_input_shape_non_overlap(output_shape, tile_shape):
        return tuple(math.ceil(output_shape[i] * tile_shape[i]) for i in range(len(tile_shape)))

    @staticmethod
    def _compute_input_shape_with_overlap(output_shape, tile_shape):
        return tuple(math.ceil(output_shape[i] + tile_shape[i] - 1) for i in range(len(tile_shape)))


class AbstractPoolingLayer(AbstractConvolutionalLayer):
    '''
    Implements a pooling layer for a neural net,
    which divides an input matrix/vector into 'tiles'
    and then performs a particular function on each tile
    to compute the layer's output.
    '''

    def __init__(self, tile_shape, func=numpy.amax, overlap_tiles=False):
        super().__init__(tile_shape, overlap_tiles)
        self.func = func

    def process(self, inputs, remember_inputs=False):
        inputs = coerce_to_3d(inputs)
        result = []
        for layer in inputs:
            result.append(self.convolve2d(layer, self.func))
        result = numpy.array(result)
        assert len(result.shape) == 3
        if remember_inputs:
            self._last_inputs.append(inputs)
            self._last_outputs.append(result)
        return result

    # TODO: implement backpropagation by remembering which cell was used for output and backpropagating to there


class FullyConnectedLayer:
    def __init__(self, num_ins, num_outs, training_rate=0.01, activation_function_name='relu'):
        self.training_rate = training_rate
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.activation_function, self.activation_function_deriv = self._get_activation_function(
            activation_function_name)
        self.weights = self._make_random_weights(num_ins, num_outs)
        self.bias = self._make_bias(num_outs)
        self._last_ins = []
        self._last_outs = []
        self._last_intermediates = []
        assert is_column_vector(self.bias)

    @staticmethod
    def _make_random_weights(num_ins, num_outs):
        return (2 * numpy.random.rand(num_outs, num_ins) - 1)  # / (num_ins * num_outs)

    @staticmethod
    def _make_bias(num_outs):
        return numpy.random.rand(num_outs, 1) / num_outs  # second dimension says num_cols = 1, so a col vector

    def process(self, inputs, remember_inputs=False):
        intermediate = self._compute_neural_output(inputs)
        results = self.activation_function(intermediate)
        if remember_inputs:
            self._last_ins.append(inputs)
            self._last_intermediates.append(intermediate)
            self._last_outs.append(results)
        return results

    def _compute_neural_output(self, raw_inputs):
        inputs = to_column_vector(raw_inputs)
        if inputs.size != self.num_ins:
            raise ValueError(
                "Fully connected layer expected %d inputs (excluding bias), found %d" % (self.num_ins, inputs.size))
        raw_output_vec = numpy.matmul(self.weights, inputs) + self.bias
        assert len(raw_output_vec) == self.num_outs
        return raw_output_vec

    def backpropagate(self, error):
        '''
        Updates any relevant weights
        and then returns the error for the previous layer.
        :param error: the error in what this layer output
        :return: the error for the previous layer
        '''
        # should be a column:
        if not is_column_vector(error):
            error = to_column_vector(error)
        assert is_column_vector(error)

        # assert intermediate_delta.shape == self._last_intermediate.shape
        last_in = self._last_ins.pop()
        last_intermediate = self._last_intermediates.pop()
        # last_out = self._last_outs.pop()

        assert is_column_vector(last_intermediate)
        assert last_intermediate.size == self.num_outs
        deriv = self.activation_function_deriv(last_intermediate)
        last_layers_error = numpy.multiply(error, deriv)
        assert is_column_vector(last_layers_error)
        assert last_layers_error.size == self.num_outs

        gradient = numpy.matmul(error, to_row_vector(last_in))
        print("        Finished backpropagation for %5d x %2d dense layer "
              "w/ wt. updates %.1E to %.1E (avg. wt. %.1E), "
              "bias updates %.1E to %.1E (avg. bias %.1E)" % (
                  self.num_ins, self.num_outs,
                  self.training_rate * numpy.min(gradient), self.training_rate * numpy.max(gradient),
                  numpy.average(self.weights),
                  self.training_rate * numpy.min(error), self.training_rate * numpy.max(error),
                  numpy.average(self.bias)))
        assert gradient.shape == self.weights.shape
        self.weights -= self.training_rate * gradient
        assert error.shape == self.bias.shape
        self.bias -= self.training_rate * error

        return numpy.matmul(numpy.transpose(self.weights), last_layers_error)

    @staticmethod
    def _get_activation_function(activation_function_name):
        if activation_function_name == "relu":
            return vec_relu, vec_relu_deriv
        if activation_function_name == 'sigmoid':
            return vec_sigmoid, vec_sigmoid_deriv
        raise ValueError("Unrecognized activation function \"%s\"" % activation_function_name)


class MaxpoolLayer(AbstractPoolingLayer):
    '''
    Performs max pooling on a matrix of input values
    as a layer in a neural net.
    Max pooling divides up the input matrix into 'tiles' or 'pools'
    and returns the maximum value for each pool as its output.
    '''

    def __init__(self, tile_shape, overlap_tiles=False):
        super().__init__(tile_shape, func=numpy.amax, overlap_tiles=overlap_tiles)


class MinpoolLayer(AbstractPoolingLayer):
    '''
    Performs max pooling on a matrix of input values
    as a layer in a neural net.
    Max pooling divides up the input matrix into 'tiles' or 'pools'
    and returns the maximum value for each pool as its output.
    '''

    def __init__(self, tile_shape, overlap_tiles=False):
        super().__init__(tile_shape, func=numpy.amin, overlap_tiles=overlap_tiles)


class MeanpoolLayer(AbstractPoolingLayer):
    '''
    Performs max pooling on a matrix of input values
    as a layer in a neural net.
    Max pooling divides up the input matrix into 'tiles' or 'pools'
    and returns the maximum value for each pool as its output.
    '''

    def __init__(self, tile_shape, overlap_tiles=False):
        super().__init__(tile_shape, func=numpy.mean, overlap_tiles=overlap_tiles)

    def backpropagate(self, error):
        assert not self._overlap_tiles  # this algo can only handle isolated tiles
        # remove remembered throughput to prevent memory leak
        if len(self._last_inputs) != 0:
            self._last_inputs.pop()
            self._last_outputs.pop()
        last_error = error
        for i in range(len(self.tile_shape)):
            last_error = numpy.repeat(last_error, repeats=self.tile_shape[i], axis=i)
        return last_error / numpy.prod(self.tile_shape)


class ConvolutionalLayer:
    def __init__(self, filter_shape, num_filters=1, training_rate=0.01):
        self.training_rate = training_rate
        self.filters = [(numpy.random.rand(*filter_shape) - 0.5) * 2 / numpy.prod(filter_shape) for _ in
                        range(num_filters)]
        if len(filter_shape) is not 2:
            raise ValueError("Filters must be 2-dimensional")
        self.filter_shape = filter_shape
        self._last_inputs = []
        self._last_outputs = []

    def backpropagate(self, error):
        '''
        Thanks to https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
        :param error:
        :return:
        '''
        last_output = self._last_outputs.pop()
        last_input = self._last_inputs.pop()
        error = error.reshape(last_output.shape)  # returns a new view, not the underlying object
        last_error = numpy.zeros(last_input.shape)  # will always be 3-D
        # weight_gradient = numpy.zeros(self.filter_weights.shape)
        (tile_height, tile_width) = self.filter_shape  # should always be 2-D
        num_input_layers = error.shape[0] // len(self.filters)
        assert num_input_layers * len(self.filters) == error.shape[0]

        # look at each output layer as a convolution of one filter and one input layer,
        # and find the total gradient for that filter for each layer
        for filter_num in range(len(self.filters)):
            weight_gradient = numpy.zeros_like(self.filters[filter_num])
            # total up the gradient for this filter for each input layer
            for input_layer_num in range(num_input_layers):  # for loops should match the relative order in process()
                for h in range(error.shape[1]):
                    for w in range(error.shape[2]):
                        last_error[input_layer_num, h:h + tile_width, w:w + tile_width] += \
                            self.filters[filter_num] * error[filter_num * num_input_layers + input_layer_num, h, w]

                # I don't know what's wrong and this is bad.
                # I think it's producing backpropagation incorrectly
                weight_gradient += convolve(last_input[input_layer_num], numpy.flip(error[filter_num]),
                                            mode='valid', method='direct')
                # we don't update the filter until we've totaled for each layer, or else the changes would affect each gradient
            self.filters[filter_num] -= self.training_rate * numpy.flip(weight_gradient)
            print("            Finished backpropagation for %dx%d Conv. layer, filter %d, "
                  "with gradient updates from %.1E to %.1E (avg. of filter is %.1E)" % (
                      self.filter_shape[0], self.filter_shape[1], filter_num,
                      self.training_rate * numpy.min(weight_gradient), self.training_rate * numpy.max(weight_gradient),
                      numpy.average(self.filters[filter_num])),
                  flush=True)

        print("        Finish backprop for Conv. layer with passthrough error from %.1E to %.1E" % (
            numpy.min(last_error), numpy.max(last_error)),
              flush=True)
        return last_error

    def process(self, inputs, remember_inputs=False):
        '''
        This only accepts 3-D input, convolves using a 2-D filter, and always gives 3-D output.

        For example, a 3x10x10 matrix convolved with 4 2x2 filters will yield a 12x9x9 output.

        2-D input is coerced to a 3-D matrix such that a 7x7 input becomes a 1x7x7 input.

        :param inputs: a 3-D matrix of (depth)x(width)x(height) or a 2-D matrix of (width)x(height) (assumed to be of depth 1)
        :param remember_inputs: true if this input should be remembered for future learning
        :return: a 3-D matrix of (depth)x(width)x(height)
        '''
        # coerce the inputs to the right dimensionality, so it can accept an NxM array as a 1xNxM array.
        inputs = coerce_to_3d(inputs)
        assert len(inputs.shape) == 3
        results = []
        for filter2d in self.filters:
            for layer2d in inputs:
                result2d = convolve(layer2d, filter2d, mode='valid', method='direct')
                results.append(result2d)
        results = numpy.array(results)
        if remember_inputs:
            self._last_inputs.append(inputs)
            self._last_outputs.append(results)
        return results


class SimpleNeuralBinaryClassifier:
    def __init__(self):
        self.layers = []

    def fit(self, X, y, batch_size=None):
        if type(y) is not numpy.ndarray:
            y = numpy.asarray(y)
        if batch_size is None or batch_size <= 0:
            if y.size > 10:
                batch_size = min(y.size // 5, 10)
            else:
                batch_size = y.size
            print("Training with batch size %d" % batch_size)
        if len(X) != y.size:
            raise ValueError("Number of samples in X must equal the number of observations in y")

        pos_nums = []
        pos_entropies = []
        neg_nums = []
        neg_entropies = []

        for batch_start in range(0, y.size, batch_size):
            print("Predicting batch starting at sample %d" % batch_start)
            for sample_num in range(batch_start, min(y.size, batch_start + batch_size)):
                yhat = self._process(X[sample_num], remember_inputs=True)
                entropy = binary_cross_entropy(yhat, y[sample_num])
                if y[sample_num] == 1:
                    pos_nums.append(sample_num)
                    pos_entropies.append(entropy)
                else:
                    neg_nums.append(sample_num)
                    neg_entropies.append(entropy)
                print("   Finished sample %3d (%8s): %.1f%% prediction melanoma (error = %.4f)" % (
                    sample_num, "melanoma" if y[sample_num] else "benign", yhat * 100, entropy), flush=True)
            print("Learning from results of prediction...", flush=True)
            for sample_num in reversed(range(batch_start, min(y.size, batch_start + batch_size))):
                print("    %d" % sample_num, flush=True)
                if y[sample_num] == 1:
                    true_outputs = to_column_vector([1])
                elif y[sample_num] == 0:
                    true_outputs = to_column_vector([0])
                else:
                    raise ValueError("Unexpected true observation %s for sample %d. "
                                     "This neural net can only perform "
                                     "binary classification." % (y[sample_num], sample_num))
                self._backpropagate(true_outputs)
            print()
        plt.plot(neg_nums, neg_entropies, 'b-', pos_nums, pos_entropies, 'r-')
        plt.legend(handles=[Patch(color='red', label='Melanoma'), Patch(color='blue', label='Not melanoma')])
        plt.show()

    def predict(self, X):
        yhat = []

        for i in range(len(X)):
            print("Making prediction for sample %d" % i)
            xrow = X[i]
            prediction = self._process(xrow, remember_inputs=False)
            yhat.append(prediction.flatten())

        yhat = numpy.asarray(yhat)
        assert yhat.shape == (len(X), 2)
        return yhat

    def _process(self, inputs, remember_inputs=True):
        for layer in self.layers:
            inputs = layer.process(inputs, remember_inputs=remember_inputs)
            print("        %s has average output %.2f" % (type(layer).__name__, numpy.average(inputs)))
        return inputs

    def _backpropagate(self, true_outputs):
        for layer in reversed(self.layers):
            true_outputs = layer.backpropagate(true_outputs)

    def add_layer(self, layer):
        self.layers.append(layer)
