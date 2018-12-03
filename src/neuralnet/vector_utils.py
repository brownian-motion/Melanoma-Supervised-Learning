import numpy


def is_column_vector(xs):
    if type(xs) is not numpy.ndarray:
        return False
    return len(xs.shape) == 2 and xs.shape[1] == 1


def to_column_vector(xs):
    if type(xs) is numpy.ndarray:
        return xs.reshape(xs.size, 1)
    return numpy.array(xs).reshape(len(xs), 1)


def is_row_vector(xs):
    if type(xs) is not numpy.ndarray:
        return False
    return len(xs.shape) == 2 and xs.shape[0] == 1


def to_row_vector(xs):
    if type(xs) is numpy.ndarray:
        return xs.reshape(1, xs.size)
    return numpy.array(xs).reshape(1, len(xs))
