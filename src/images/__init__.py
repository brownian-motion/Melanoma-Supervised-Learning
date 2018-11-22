import json
from os import listdir
from os.path import isfile, join

import cv2


class Sample:
    def __init__(self, name, diagnosis, image_dim):
        self.image_dim = image_dim
        self.name = name
        self.diagnosis = diagnosis

    def get_image(self):
        '''
        Returns a NumPy matrix of RGB triplets.
        :return:
        '''
        return cv2.imread(self.name + ".jpg")

    def get_grayscale_image(self):
        '''
        Returns a NumPy matrix of grayscale valeus for each pixel
        :return:
        '''
        return cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _get_metadata_file_names(dirname="ISIC-images/UDA-1"):
        return [f for f in listdir(dirname) if isfile(join(dirname, f)) and f.endswith("json")]

    @staticmethod
    def get_samples(dirname="ISIC-imagse/UDA-1"):
        return [Sample.load_sample(open(filename)) for filename in Sample._get_metadata_file_names(dirname)]

    @staticmethod
    def load_sample(file):
        sample_json = json.load(file)
        acquisition_data = sample_json[u'meta'][u'acquisition']
        return Sample(
            sample_json[u'name'],
            sample_json[u'meta'][u'clinical'][u'diagnosis'],
            (acquisition_data[u'pixelsX'], acquisition_data[u'pixelsY'])
        )
