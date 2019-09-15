import numpy as np
import random
import warnings
import os

import keras
from ..utils.image import read_image_bgr

from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)

voc_classes_sub = {
    'person'      : 0,
    'cat'         : 1,
    'chair'       : 2,
    'dog'         : 3,
    'sofa'        : 4
}

class PredictImageGenerator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        imgDir,
        transform_generator = None,
        classes=voc_classes_sub,
        shuffle=False,
        image_extension='.jpg',
        image_min_side=800,
        image_max_side=1333,
    ):
        self.transform_generator    = transform_generator
        self.shuffle                = shuffle
        self.image_extension        = image_extension
        self.classes                = classes
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.imgDir = imgDir
        self.list_IDs = self._get_list_ids()

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.list_IDs)

    def get_file_name(self, idx):
        return self.list_IDs[idx]

    def _get_list_ids(self):
        ret = []
        name_list = os.listdir(self.imgDir)
        for temp_name in name_list:
            temp_name_noext, temp_ext = os.path.splitext(temp_name)
            ret.append(temp_name_noext)
        return ret

    def size(self):
        return len(self.list_IDs)

    def num_classes(self):
        return len(self.classes)

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path  = os.path.join(self.imgDir, self.list_IDs[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = os.path.join(self.imgDir, self.list_IDs[image_index] + self.image_extension)
        return read_image_bgr(path)

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        return self.load_image(index)
