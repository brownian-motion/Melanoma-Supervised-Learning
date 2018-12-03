import unittest

from neuralnet.vector_utils import *


def get_sample_column():
    return numpy.array([1, 2, 3, 4]).reshape(4, 1)


def get_sample_row():
    return numpy.array([1, 2, 3, 4]).reshape(1, 4)


class MatrixShapeTest(unittest.TestCase):
    def test_column_detection(self):
        self.assertTrue(is_column_vector(get_sample_column()), msg="Could not detect column vector")

    def test_row_is_not_column(self):
        self.assertFalse(is_column_vector(get_sample_row()), msg="Row should not be a column")

    def test_row_detection(self):
        self.assertTrue(is_row_vector(get_sample_row()), msg="Could not detect row vector")

    def test_column_is_not_row(self):
        self.assertFalse(is_row_vector(get_sample_column()), msg="Column should not be row")

    def test_can_make_column_from_list(self):
        col = to_column_vector([1, 2, 3, 4])
        self.assertTrue(is_column_vector(col), msg="Did not make column vector correctly: %s" % col)

    def test_can_make_row_from_list(self):
        row = to_row_vector([1, 2, 3, 4])
        self.assertTrue(is_row_vector(row), msg="Did not make row vector correctly: %s" % row)

    def test_can_make_column_from_row(self):
        col = to_column_vector(get_sample_row())
        self.assertTrue(is_column_vector(col), msg="Did not make column vector from row correctly: %s" % col)

    def test_can_make_row_from_column(self):
        row = to_row_vector(get_sample_column())
        self.assertTrue(is_row_vector(row), msg="Did not make row vector from column correctly: %s" % row)
