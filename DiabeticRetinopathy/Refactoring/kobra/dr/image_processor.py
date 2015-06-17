import numpy as np
import pandas as pd
import cv2
from image_reader import ImageReader

class ImageProcessor(object):
    '''
    Base class for all of retina image processing
    '''
    def __init__(self, root, im_file, masks_dir):
        self._reader = ImageReader(root, im_file, masks_dir)
        self._root = root

        self._image = self._reader.image
        self._mask = self._reader.mask

    @property
    def image(self):
        return self._image

    @property
    def mask(self):
        return self._mask

