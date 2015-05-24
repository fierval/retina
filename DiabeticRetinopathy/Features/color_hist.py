import numpy as np
import cv2
import environment as env


class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask = None):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)

        return hist.flatten()