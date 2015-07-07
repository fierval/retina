import cv2
import pandas as pd
import numpy as np
from kobra.imaging import show_images
from image_reader import ImageReader
from os import path
from imutils import translate


class DetectOD(object):
    '''
    Detecting location of the optic disk (OD)
    based on the morphological method proposed in
    Suero, Marin, et. al. -- Locating the Optic Disc in Retinal Images Using
    Morphological Techniques.
    
    Image masks should be extracted prior to using this class and placed in mask_dir.
    These masks are detected locations of the actual eye in the image being processed 
    '''
    def __init__(self, root, im_file, mask_dir):

        self._reader = ImageReader(root, im_file, mask_dir)
        self._img = self._reader.image
        self._orig_mask = self._reader.mask

        self._scale = np.float32(self._orig_mask.shape) / 540.
        self._scale[0], self._scale[1] = self._scale[1], self._scale[0]
                
        self._img = cv2.resize(self._img, (540, 540))
        
        self._mask = cv2.resize(self._orig_mask, (540, 540))

        self._shifted = cv2.bitwise_and(translate(self._mask, 40, 30), translate(self._mask, -40, -30))

        self._img [self._mask == 0] = 0

        # get green channel
        self._green = self._img[:, :, 1].astype('float32')
    
        # kick-start processing
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

    @property
    def scale(self):
        return self._scale

    def remove_light_reflex(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(self._processed, cv2.MORPH_OPEN, kernel)
    
    def shade_correct(self):
        gamma = self.remove_light_reflex()

        # mean filter
        self._processed = cv2.medianBlur(self._processed, 3)
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

    def locate_disk(self, rescale = True):
        self.shade_correct()
        self.apply_morphology()

        pr = self._processed.copy().astype('uint8')

        # kill possible bright edges of the image
        pr[self._shifted == 0] = 0

        _, maxCol, _, ctr = cv2.minMaxLoc(pr)
        if rescale:
            x, y = self._rescale_to_original_mask(ctr)
            ctr = (int(x), int(y))

        return ctr

    def show_detected(self, ctr):
        pr = self._processed.copy().astype('uint8')
        cv2.circle(pr, ctr, 50, 0, 3)
        cv2.circle(self.image, ctr, 50, (0, 0, 0), 3)
        show_images([self.image, pr])

    def _rescale_to_original_mask(self, pt):
        ctr_orig = np.float32(pt) * self._scale
        return (ctr_orig[0], ctr_orig[1])

    def mask_off_od(self):
        ctr_mask = self.locate_disk()
        mask = self._orig_mask
        mask_radius = sqrt(cv2.countNonZero(mask) / math.pi)
        od_radius = int(round(0.22 * mask_radius))

        cv2.circle(mask, ctr_mask, od_radius, 0, -1)
        return mask

    def show_detected_mask(self):
        mask = self.mask_off_od()
        mask = cv2.resize(mask, (540, 540))
        im = self._img.copy()
        im[mask == 0] = 0
        show_images([self._img, im])