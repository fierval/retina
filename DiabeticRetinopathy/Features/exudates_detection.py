import numpy as np
import pandas as pd
from kobra.imaging import show_images, salt_and_peper
import imutils
import cv2
from os import path

class DetectExudates(object):
    '''
    Exudates detection
    '''

    def __init__(self, im_file, mask_dir):

        assert (path.exists(im_file)), "Image not found"
        assert (path.exists(mask_dir)), "Mask not found"

        self._im_name = path.splitext(path.split(im_file)[1])[0]
        mask_file = path.join(mask_dir, self._im_name + ".png")
        self._img = cv2.imread(im_file)
        self._mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        assert (self._img.size > 0), "Image not found"
        assert (self._mask.size > 0), "Mask not found"
        
        # rescale to 256 x 256
        self._img = cv2.resize(self._img, (256, 256))
        self._mask = cv2.resize(self._mask, (256, 256))

        self._img [self._mask == 0] = 0

        # intensity image
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HLS)

        # HLS space, intensity is the second channel
        self._intensity = self._img[:, :, 1]

        self._processed = self._intensity.copy()

    def preprocess(self):
        # median blur with a 3x3 kernel
        self._processed = cv2.blur(self._processed, (3, 3))

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit = 3., tileGridSize = (4, 4))
        self._processed = clahe.apply(self._processed)
        