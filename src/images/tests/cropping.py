import unittest

from images.cropping import *


class CroppingTest(unittest.TestCase):
    def test_cropping_wide_mat_to_square_returns_square(self):
        inp = numpy.zeros((20, 25))
        out = crop_to_centered_square(inp)
        self.assertEqual((20, 20), out.shape)

    def test_cropping_tall_mat_to_square_returns_square(self):
        inp = numpy.zeros((26, 20))
        out = crop_to_centered_square(inp)
        self.assertEqual((20, 20), out.shape)

    def test_cropping_3d_matrix_preserves_3rd_dimension_length(self):
        inp = numpy.zeros((15, 12, 3))
        out = crop_to_centered_square(inp)
        self.assertEqual((12, 12, 3), out.shape)

    def test_cropping_square_returns_same_mat(self):
        inp = numpy.random.rand(13, 13, 2)
        out = crop_to_centered_square(inp)
        self.assertTrue(numpy.equal(inp, out).all(), msg="Cropping square matrix should yield an identical matrix")
