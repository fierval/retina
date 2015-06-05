import cv2
import pandas as pd
import numpy as np
from kobra.imaging import show_images
from os import path
from imutils import translate

class DetectOD(object):
    def __init__(self, im_file, mask_dir):

        assert (path.exists(im_file)), "Image not found"
        assert (path.exists(mask_dir)), "Mask not found"

        self._im_name = path.splitext(path.split(im_file)[1])[0]
        mask_file = path.join(mask_dir, self._im_name + ".png")
        self._img = cv2.imread(im_file)
        self._mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        assert (self._img.size > 0), "Image not found"
        assert (self._mask.size > 0), "Mask not found"

        #self._img = kim.pyr_blurr(self._img)
        self._img = cv2.resize(self._img, (540, 540))
        self._mask = cv2.resize(self._mask, (540, 540))

        self._shifted = cv2.bitwise_and(translate(self._mask, 50, 0), translate(self._mask, -50, 0))

        self._img [self._mask == 0] = 0

        # get green channel
        self._green = self._img[:, :, 1].astype('float32')

        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._processed = self._green.copy()

    @property
    def image(self):
        return self._img

    @property
    def active_channel(self):
        return self._green

    @property 
    def processed(self):
        return self._processed

    def remove_light_reflex(self):
        return cv2.morphologyEx(self._processed, cv2.MORPH_OPEN, self._kernel)
    
    def shade_correct(self):
        gamma = self.remove_light_reflex()

        # mean filter
        self._processed = cv2.blur(self._processed, (3, 3))
        # Gaussian convolution
        self._processed = cv2.GaussianBlur(self._processed, (9, 9), 1.8)
        
        # further averaging
        # before mean blurring, set black regions of the image to the average gray value
        self._processed[self._mask == 0] = cv2.mean(self._processed, self._mask)[0]

        background = cv2.blur(self._processed, (69, 69))

        # distance matrix between background and the image
        dist = gamma - background
        minDist = np.min(dist)
        interval = np.max(dist) - minDist
        self._processed = np.around((dist - minDist) * 255. / interval)
        self._processed[self._mask == 0] = 0

        return self._processed
    
    def apply_morphology(self):
        for r in range(2, 12, 2):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r, 2 * r))
            self._processed = cv2.morphologyEx(self._processed, cv2.MORPH_OPEN, kernel)
            self._processed = cv2.morphologyEx(self._processed, cv2.MORPH_CLOSE, kernel)

        return self._processed

    def locate_disk(self):
        self.shade_correct()
        self.apply_morphology()

        pr = self._processed.copy().astype('uint8')

        # kill possible bright edges of the image
        pr[self._shifted == 0] = 0

        _, maxCol, _, ctr = cv2.minMaxLoc(pr)
        return ctr

    def show_detected(self, ctr):
        pr = self._processed.copy().astype('uint8')
        cv2.circle(pr, ctr, 50, 0, 3)
        cv2.circle(self.image, ctr, 50, (0, 0, 0), 3)
        show_images([self.image, pr])