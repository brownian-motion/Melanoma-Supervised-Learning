# based on stack ex question 8847 - feature extraction in python
# need to setup a loop to save these to csv file

from PIL import Image, ImageFilter
import matplotlib
import numpy
from pylab import *
import csv

# image = Image.open('img.jpg')
# im = image.convert('L')
# img_arr = numpy.array(image)
# figure()
# gray()
# contour(im, origin='image')
# im2 = im.filter(ImageFilter.CONTOUR)
# img_arr = numpy.array(im2)

# print(hist(img_arr.flatten(), 128))
#show()

# features_list = []
# features_list.append(hist(img_arr.flatten(), 128))
# a = numpy.array(features_list)
# numpy.savetxt("melanoma_data.csv", a, delimiter=',')

# hist = hist(img_arr.flatten(), 128)
# features = []
# for x in range(0,128):
#     features.append(hist[0][x])



features = []
count = 1

import glob
import json
for filename in glob.glob('UDA-1/*.json'):
    with open(filename) as data_file:
        data_loaded = json.load(data_file)
        img_path = 'UDA-1/'
        img_path += data_loaded['name']
        img_path += '.jpg'

        # process image
        image = Image.open(img_path)
        im = image.convert('L')
        im2 = im.filter(ImageFilter.CONTOUR)
        img_arr = numpy.array(im2)
        img_hist = hist(img_arr.flatten(), 128)

        # vectorize
        feature_row = []
        for x in range(0,128):
            feature_row.append(img_hist[0][x])

        diag = data_loaded["meta"]["clinical"]["benign_malignant"]
        if diag == "benign":
            feature_row.append('0')
        else:
            feature_row.append('1')

        #features.append(feature_row)

        with open('melanoma_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow(feature_row)

        print(count)
        count += 1
        print(img_path)

# with open('melanoma_data.csv', 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile,delimiter=',')
#     for row in features:
#         writer.writerow(row)
    
