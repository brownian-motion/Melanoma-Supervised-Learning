import math

import numpy


def scalar_relu_deriv(x):
    return 1 if x > 0 else 0


vec_relu_deriv = numpy.vectorize(scalar_relu_deriv)


def scalar_relu(x):
    return max(0, x)


def vec_relu(xs):
    return numpy.maximum(0, xs)


def vec_softmax(xs):
    xs -= numpy.min(xs)  # softmax is invariant to constant offsets, this lets us renormalize huge exponents
    exps = numpy.array([math.exp(x) for x in xs])
    return exps / sum(exps)


def vec_softmax_deriv(xs):
    raise NotImplementedError("d softmax / d x is not implemented")


def scalar_sigmoid(x):
    if x > 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp = math.exp(x)
        return exp / (exp + 1)


def vec_sigmoid(xs):
    return 1 / (1 + numpy.exp(-xs))


def vec_sigmoid_2(xs):
    """
    just like vec_sigmoid(), but ranging from -1 to 1
    """
    return vec_sigmoid(xs) * 2 - 1


def scalar_sigmoid_deriv(x):
    sigmoid = scalar_sigmoid(x)
    return (sigmoid - 1) * sigmoid


vec_sigmoid_deriv = numpy.vectorize(scalar_sigmoid_deriv)


def vec_sigmoid_2_deriv(xs):
    return 2 * vec_sigmoid_deriv(xs)


def vec_identity(xs):
    return xs


def vec_identity_deriv(xs):
    return numpy.full_like(xs, 1)
