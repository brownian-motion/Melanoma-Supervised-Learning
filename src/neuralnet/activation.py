import math

import numpy


def scalar_relu_deriv(x):
    return 1 if x > 0 else 0


def vec_relu_deriv(xs):
    return map_function_numpy(scalar_relu_deriv, xs)


def scalar_relu(x):
    return max(0, x)


def vec_relu(xs):
    return map_function_numpy(scalar_relu, xs)


def vec_softmax(xs):
    exps = numpy.array([math.exp(x) for x in xs])
    return exps / sum(exps)


def vec_softmax_deriv(xs):
    raise NotImplementedError("d softmax / d x is not implemented")


def map_function_numpy(func, vec):
    if type(vec) is not numpy.ndarray:
        vec = numpy.array(vec)
    return numpy.fromiter((func(out) for out in vec), vec.dtype).reshape(vec.shape)
