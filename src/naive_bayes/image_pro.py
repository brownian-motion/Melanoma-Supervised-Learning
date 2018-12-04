# based on stack ex question 8847 - feature extraction in python
# need to setup a loop to save these to csv file

from PIL import Image
import matplotlib
import numpy
from pylab import *

image = Image.open('img.jpg')
im = image.convert('L')
img_arr = numpy.array(image)
figure()
gray()
contour(im, origin='image')
hist(img_arr.flatten(), 128)
show()