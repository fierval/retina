import numpy as np
import pandas as pd
from kobra.imaging import show_images, salt_and_peper
import imutils
import cv2
from os import path
from od_detection import DetectOD

class DetectExudates(object):
    '''
    Exudates detection
    '''

    def __init__(self, im_file, mask_dir):
        self._od_detector = DetectOD(im_file, mask_dir)

        # rescale to 256 x 256
        self._img = cv2.resize(self._od_detector._img, (256, 256))
        # mask off OD
        self._orig_mask = self._od_detector.mask_off_od()
        self._mask = cv2.resize(self._orig_mask, (256, 256))

        # intensity image
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HLS)

        # HLS space, intensity is the second channel
        self._intensity = self._img[:, :, 1]

        self._processed = self._intensity.copy()

    def preprocess(self):
        # salt & pepper noise at 1%
        self._processed = salt_and_peper(self._processed)

        # median blur with a 3x3 kernel
        self._processed = cv2.medianBlur(self._processed, 3)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit = 3., tileGridSize = (4, 4))
        self._processed = clahe.apply(self._processed)
        
