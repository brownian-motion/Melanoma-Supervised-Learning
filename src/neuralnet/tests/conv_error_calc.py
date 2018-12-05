"""
This file has a unit test that just confirms that
a naive implementation and a fast, convolve() implementation
for calculating backpropagated error through a neural net
are equivalent.
"""

import timeit
import unittest

import numpy
from scipy.signal import convolve  # to perform convolution


def perform_original_error_calc(filter, error_layer):
    (tile_height, tile_width) = filter.shape
    error_calc_start_time = timeit.default_timer()
    for _ in range(100):
        last_error = numpy.zeros((error_layer.shape[0] + tile_height - 1, error_layer.shape[1] + tile_width - 1))
        for (h, w) in numpy.ndindex(*error_layer.shape):
            last_error[h:h + tile_height, w:w + tile_width] += \
                filter * error_layer[h, w]
    error_calc_duration = timeit.default_timer() - error_calc_start_time

    return last_error, error_calc_duration


def perform_optimized_error_calc(filter, error_layer):
    start = timeit.default_timer()
    for _ in range(100):
        opt_result = convolve(error_layer, filter)
        dur = timeit.default_timer() - start
    return opt_result, dur


class ErrorCalcTest(unittest.TestCase):
    def test_calcs_return_same_result(self):
        filter = numpy.random.rand(5, 5)
        error_layer = numpy.random.rand(130, 130)

        original_result, orig_dur = perform_original_error_calc(filter, error_layer)
        opt_result, opt_dur = perform_optimized_error_calc(filter, error_layer)
        self.assertSequenceEqual(original_result.shape, opt_result.shape)
        for (r, c) in numpy.ndindex(*original_result.shape):
            self.assertAlmostEqual(original_result[r, c], opt_result[r, c],
                                   msg=("Results differ at [%d, %d]" % (r, c)), places=6)
        self.assertLess(opt_dur, orig_dur, msg="Optimized algo should be faster than original method")
        print("Sped up by %.1f times (%.2E opt vs %.2E orig)" % (orig_dur / opt_dur, opt_dur, orig_dur))
