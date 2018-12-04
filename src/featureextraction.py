from pylab import *
from PIL import Image, ImageFilter
from PIL import *
from images import Sample
import matplotlib


class FeatureExtractor:
    def __init__(self):
        features_list = ""

def make_extractor():
    return FeatureExtractor

def get_features(im_arr):
    """This method returns an array of features, divided
    by four pixels of darkness, into a total of 64 features."""
    features_list = np.histogram(im_arr, bins=64)
    return features_list


def convert_image(sample):
    """
    This method will specifically take an image and convert it to
    Greyscale before returning an image. """
    # Import an image
    image = sample.get_grayscale_image()
    return image

def get_image_array(image):
    """This method takes an image and returns an array representation of the image called im_arr"""
    im_arr = array(image)
    return im_arr

def get_blur_of_converted_image(im_arr):
    """This method takes an array representation of an image creates an Guassian Blur of the input image."""
    figure()
    im = Image.fromarray(im_arr)
    p = im.convert("L").filter(ImageFilter.GaussianBlur(radius=2))
    p.show()

def get_flatten_converted_image(image):
    """This method takes an image as input, and create a figure that displays a flat, density distribution of the picture."""
    # create a new figure
    figure()
    gray()
    # show contours with origin upper left corner
    contour(image, origin='image')
    axis('equal')
    axis('off')

def get_darkness_distribution(im_array):
    """This method takes im_arr, and records the darkness distribution in a histogram."""
    figure()
    hist(im_array.flatten(), 128)
    show()