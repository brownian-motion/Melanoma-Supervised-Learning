import math

import numpy


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
        exps = numpy.array([math.exp(inp) for inp in inputs])
        return exps / sum(exps)


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
        output_shape = self._compute_output_shape(input.shape, self.tile_shape)
        output = numpy.zeros(output_shape)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                tile = self._get_tile((i, j), input, self.tile_shape)
                output[i, j] = self.func(tile)
        return output


def relu(x):
    return max(0, x)


def relu_deriv(x):
    return 1 if x > 0 else 0


class FullyConnectedLayer:
    def __init__(self, training_rate, num_ins, num_outs, activation_function_name='relu',
                 initial_weight=None):
        self.training_rate = training_rate
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.activation_function, self.activation_function_deriv = self._get_activation_function(
            activation_function_name)
        self.weights = self._make_weights(num_ins, num_outs, initial_weight)

    @staticmethod
    def _make_weights(num_ins, num_outs, initial_weight=None):
        if initial_weight is None:
            initial_weight = 1 / (num_ins + 1)
        return numpy.full((num_outs, 1 + num_ins), initial_weight)

    @staticmethod
    def _append_bias_to_inputs(input_vec):
        return numpy.append(input_vec, 1)

    def process(self, inputs):
        raw_output_vec = self._compute_neural_output(inputs)
        return map_function_numpy(self.activation_function, raw_output_vec)

    def _compute_neural_output(self, raw_inputs):
        raw_inputs = numpy.array(raw_inputs).flatten()
        if len(raw_inputs) != self.num_ins:
            raise ValueError(
                "Fully connected layer expected %d inputs (excluding bias), found %d" % (self.num_ins, len(raw_inputs)))
        raw_inputs = self._append_bias_to_inputs(raw_inputs)
        raw_output_vec = numpy.matmul(self.weights, numpy.transpose(raw_inputs))
        assert len(raw_output_vec) == self.num_outs
        return raw_output_vec

    def back_propagate(self, inputs, outputs, correct_outputs):
        neural_intermediate = self._compute_neural_output(inputs)
        sigma = (outputs * numpy.transpose(self.weights)).multiply(self.activation_function_deriv(neural_intermediate))
        gradient = numpy.transpose(inputs) * sigma
        raise NotImplementedError("Haven't implemented back-propagation yet")

    @staticmethod
    def _get_activation_function(activation_function_name):
        if activation_function_name == "relu":
            return relu, relu_deriv
        raise ValueError("Unrecognized activation function \"%s\"" % activation_function_name)


def map_function_numpy(func, vec):
    return numpy.fromiter((func(out) for out in vec), vec.dtype, len(vec))


class MaxpoolLayer(AbstractPoolingLayer):
    '''
    Performs max pooling on a matrix of input values
    as a layer in a neural net.
    Max pooling divides up the input matrix into 'tiles' or 'pools'
    and returns the maximum value for each pool as its output.
    '''

    def __init__(self, tile_shape, overlap_tiles=False):
        super().__init__(tile_shape, func=numpy.amax, overlap_tiles=overlap_tiles)
