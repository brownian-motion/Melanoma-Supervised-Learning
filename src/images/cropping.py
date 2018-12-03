import numpy


def crop_to_centered_square(mat):
    '''
    Crops the given matrix (using a NumPy view) on the first two axes
    and returns a view into the center that is square.
    For example, if given a 1000x1200x3 matrix,
    this method returns a 1000x1000x3 matrix by 'cropping'
    100 columns in each direction of the center on the second axis.
    :param mat: a matrix (of at least 2 dimensions) to crop
    :return: a view into that matrix that is cropped so that the first two axes are of equal size
    '''
    if type(mat) != numpy.ndarray:
        raise ValueError("I only know how to crop NumPy arrays.")
    if len(mat.shape) < 2:
        raise ValueError("I can only crop into a square if the matrix is 2-D or greater!")
    if (mat.shape[0] > mat.shape[1]):
        top = (mat.shape[0] - mat.shape[1]) // 2
        bottom = top + mat.shape[1]
        return mat[top:bottom, :]
    else:
        left = (mat.shape[1] - mat.shape[0]) // 2
        right = left + mat.shape[0]
        return mat[:, left:right]
