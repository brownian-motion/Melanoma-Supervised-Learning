import json
import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy

from images.cropping import crop_to_centered_square

STANDARD_IMAGE_LENGTH = 768


class Sample:
    def __init__(self, name, diagnosis, image_dim, filename, num_rotations=0, flip_image=False):
        self.filename = filename
        self.image_dim = image_dim
        self.name = name
        self.diagnosis = diagnosis
        self._num_rotations = num_rotations  # for upsampling
        self._do_flip = flip_image  # for upsampling

    def get_image(self):
        '''
        Returns a square NumPy matrix of the RGB triplets in the central square of this Sample's image.
        For example, if this image is 1000x760, then it returns a 760x760x3 matrix.
        :return: a square NumPy matrix of the RGB triplets in the central square of this Sample's image.
        '''
        image_raw = self.get_image_raw()
        image = crop_to_centered_square(image_raw)
        image = cv2.resize(image, (STANDARD_IMAGE_LENGTH, STANDARD_IMAGE_LENGTH))
        image = numpy.rot90(image, self._num_rotations)
        if self._do_flip:
            image = numpy.fliplr(image)
        return image

    def get_image_raw(self):
        return cv2.imread(self.filename)

    def get_grayscale_image(self):
        '''
        Returns a NumPy matrix of grayscale values for each pixel
        :return:
        '''
        return cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _get_metadata_file_names(dirname):
        return [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f)) and f.endswith("json")]

    @staticmethod
    def get_samples(dirname):
        return [Sample.load_sample(open(filename)) for filename in Sample._get_metadata_file_names(dirname)]

    @staticmethod
    def get_upsampled_copies(samples):
        '''
        Returns 8 copies of the current sample,
        made by rotating 0 to 4 times,
        and either flipping or not flipping.
        :param samples: some Sample objects to upsample
        :return: many duplicates of the given Samples
        '''
        samples = [sample.get_rotated_copy(num_rotations) for num_rotations in [0, 1, 2, 3] for sample in samples]
        samples += [sample.get_flipped_copy() for sample in samples]
        return samples

    def get_rotated_copy(self, num_rotations):
        from copy import copy
        sample_copy = copy(self)
        sample_copy._num_rotations = num_rotations
        return sample_copy

    def get_flipped_copy(self):
        from copy import copy
        sample_copy = copy(self)
        sample_copy._do_flip = not self._do_flip
        return sample_copy

    @staticmethod
    def load_sample(file):
        sample_json = json.load(file)
        acquisition_data = sample_json[u'meta'][u'acquisition']
        return Sample(
            sample_json[u'name'],
            sample_json[u'meta'][u'clinical'][u'diagnosis'],
            (acquisition_data[u'pixelsX'], acquisition_data[u'pixelsY']),
            os.path.dirname(file.name) + "/" + sample_json[u'name'] + ".jpg"
        )
