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


class AbstractNonOverlapPoolingLayer:
    def __init__(self, tile_shape, func=numpy.amax):
        if len(tile_shape) != 2:
            raise ValueError(
                "This pooling layer implementation only knows how to handle 2-D tiles, inputs, and outputs")
        self.tile_shape = tile_shape
        # calculate the number of non-overlapping tiles
        self.func = func

    def _compute_output_shape(self, input_shape, tile_shape):
        return tuple(math.ceil(input_shape[i] / tile_shape[i]) for i in range(len(tile_shape)))

    def process(self, input):
        output_shape = self._compute_output_shape(input.shape, self.tile_shape)
        output = numpy.zeros(output_shape)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                tile = self._get_tile((i, j), input, self.tile_shape)
                output[i, j] = self.func(tile)
        return output

    def _get_tile(self, tile_coord, input, tile_shape):
        if len(tile_shape) != 2 or len(tile_coord) != 2:
            raise ValueError("Sorry, this convolutional layer only knows how to get 2-D tiles from input matrices")
        left = tile_coord[0] * tile_shape[0]
        right = left + tile_shape[0]
        top = tile_coord[1] * tile_shape[1]
        bottom = top + tile_shape[1]
        output = input[left:right, top:bottom]
        assert output.shape == tile_shape
        return output


class NonOverlapMaxpoolLayer(AbstractNonOverlapPoolingLayer):
    def __init__(self, tile_shape):
        AbstractNonOverlapPoolingLayer.__init__(self, tile_shape, func=numpy.amax)
