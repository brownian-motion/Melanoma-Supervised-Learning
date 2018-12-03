from math import log


def binary_cross_entropy(yhat, y):
    if y == 1 or y is True:
        if yhat == 0:
            return 10  # equivalent to an error of 10^-10
        return -log(yhat)
    elif y == 0 or y is False:
        if yhat == 1:
            return 10  # equivalent to an error of 10^-10
        return -log(1 - yhat)
    else:
        raise ValueError("Unexpected actual (measured) value for binary output: %s" % y)
