from neuralnet.activation import *
from neuralnet.vector_utils import *


class SoftmaxLayer:
    '''
    Implements a softmax neural network layer.
    This normalizes input values to the range [0,1],
    such that the sum of the outputs add to 1.
    '''

    def process(self, inputs):
        '''
        Performs a softmax normalization on the given inputs.
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
        self._last_inputs = inputs
        self._last_outputs = to_column_vector(vec_softmax(inputs))
        assert is_column_vector(self._last_outputs)
        assert self._last_outputs.size == inputs.size
        return self._last_outputs

    def backpropagate(self, true_ys):
        '''
        Updates any relevant weights (none, for this simple activation layer)
        and then returns the error for the previous layer.
        :param true_ys: the error in what this layer output
        :return: the error for the previous layer
        '''
        return true_ys - self._last_outputs


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
        self._last_input = None
        self._last_output = None

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

    def convolve(self, input_mat, convolution_func):
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

    def process(self, input):
        self._last_input = input
        self._last_output = self.convolve(input, self.func)
        return self._last_output


class FullyConnectedLayer:
    def __init__(self, num_ins, num_outs, training_rate=0.01, activation_function_name='relu'):
        self.training_rate = training_rate
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.activation_function, self.activation_function_deriv = self._get_activation_function(
            activation_function_name)
        self.weights = self._make_random_weights(num_ins, num_outs)
        self.bias = self._make_bias(num_outs)
        assert is_column_vector(self.bias)

    @staticmethod
    def _make_random_weights(num_ins, num_outs):
        return numpy.random.rand(num_outs, num_ins)

    @staticmethod
    def _make_bias(num_outs):
        return numpy.random.rand(num_outs, 1)  # second dimension says num_cols = 1, so a col vector

    def process(self, inputs):
        self._last_ins = inputs
        self._last_intermediate = self._compute_neural_output(inputs)
        self._last_outputs = self.activation_function(self._last_intermediate)
        return self._last_outputs

    def _compute_neural_output(self, raw_inputs):
        inputs = to_column_vector(raw_inputs)
        if inputs.size != self.num_ins:
            raise ValueError(
                "Fully connected layer expected %d inputs (excluding bias), found %d" % (self.num_ins, len(raw_inputs)))
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
        assert is_column_vector(self._last_intermediate)
        assert self._last_intermediate.size == self.num_outs
        deriv = self.activation_function_deriv(self._last_intermediate)
        last_layers_error = numpy.multiply(error, deriv)
        assert is_column_vector(last_layers_error)
        assert last_layers_error.size == self.num_outs

        gradient = numpy.matmul(error, to_row_vector(self._last_ins))
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


class ConvolutionalLayer(AbstractPoolingLayer):
    def __init__(self, filter_shape, training_rate=0.01):
        super().__init__(filter_shape, func=lambda tile: self.calculate_pixel_output(tile), overlap_tiles=True)
        self.training_rate = training_rate
        self.filter_weights = numpy.random.rand(*filter_shape)

    def calculate_pixel_output(self, input_tile):
        assert input_tile.shape == self.filter_weights.shape
        return numpy.sum(input_tile * self.filter_weights)

    def backpropagate(self, error):
        error = error.reshape(self._last_output.shape)  # returns a new view, not the underlying object
        last_error = numpy.zeros(self._compute_input_shape(error.shape, self.tile_shape))
        weight_gradient = numpy.zeros(self.tile_shape)
        (tile_height, tile_width) = self.tile_shape

        # can't really use the convolve function here because it moves too many tiles at once, and updates local vars
        for h in range(error.shape[0]):
            for w in range(error.shape[1]):
                last_error[h:h + tile_width, w: w + tile_width] += self.filter_weights * error[h, w]
                weight_gradient += self._last_input[h:h + tile_height, w:w + tile_width] * error[h, w]

        self.filter_weights -= self.training_rate * weight_gradient

        return last_error


class LinearNeuralNetwork:
    def __init__(self):
        self.layers = []

    def process(self, inputs):
        for layer in self.layers:
            inputs = layer.process(inputs)
        return inputs

    def backpropagate(self, true_outputs):
        for layer in reversed(self.layers):
            true_outputs = layer.backpropagate(true_outputs)

    def add_layer(self, layer):
        self.layers.append(layer)
