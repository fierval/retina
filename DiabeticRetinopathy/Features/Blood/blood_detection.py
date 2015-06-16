import numpy as np
import pandas as pd
import cv2
from os import path
import os
import shutil
from kobra.imaging import *
import pywt
import mahotas as mh
from kobra import ImageReader

root = '/kaggle/retina/train/sample/split'
im_file = '3/27224_right.jpeg'
masks_dir = '/kaggle/retina/train/masks'


class ExtractBloodVessels(object):
    '''
    Based on: http://ictactjournals.in/paper/IJSCPaper_1_563to575.pdf
    '''
    def __init__(self, root, im_file, masks_dir):
        self._reader = ImageReader(root, im_file, masks_dir)

        # keep green channel
        self._image = self._reader.image[:, :, 1].copy()
        self._mask = self._reader.mask

    def preprocess(self):
        im = self._image
        mask = self._mask

        # green channel & remove light reflex
        im_gray = remove_light_reflex(im)

        # CLAHE
        clahe = cv2.createCLAHE(3, (7, 7))
        im_clahe = clahe.apply(im_gray)

        # Haar transform
        wp = pywt.WaveletPacket2D(im_clahe, wavelet = 'haar', maxlevel = 1, mode = 'sym')
        wp.decompose()
        im_haar = wp['a'].data

        # Matched filter response
        K = matched_filter_kernel(31, 7)
        kernels = createMatchedFilterBank(K, 12)
        im_matched = applyFilters(im_haar, kernels)

        # Gabor filter response
        kernels = gabor_filters(13, n = 12, sigma = 4.5, lmbda = 10.0)
        im_matched = applyFilters(im_matched, kernels)
        mask = cv2.resize(mask, (im_matched.shape[1], im_matched.shape[0]))
        im_norm = cv2.normalize(im_matched, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype('uint8')
        im_norm [mask == 0] = 0

        # show the results
        show_images([self._reader.image])
        show_images([im_gray, im_haar], titles = ["gray", "haar"])
        show_images([im_norm], titles = ["filtered"])

        return im_norm

    def extract_blood_vessels(self, im_norm):
        mask = self._reader.rescale_mask(im_norm)
        thresh, _, _, _ = cv2.mean(im_norm, mask)

        # computes vessel regions with mahotas distance function
        Bc = np.ones((9, 9))
        threshed = (im_norm > thresh)
        distances = mh.stretch(mh.distance(threshed))

        maxima = mh.regmax(distances, Bc = Bc)

        # create watersheds
        spots,n_spots = mh.label(maxima, Bc=Bc)
        surface = (distances.max() - distances)
        im = mh.cwatershed(surface, spots)
        im *= threshed

        # label and remove the region of the largest size
        markers, n_markers = mh.label(im)

        # sizes without the background region
        sizes = mh.labeled.labeled_size(markers)[1:]

        # vessels region is the one with the largest area
        # since we have removed background region "0", add 1
        vessels_label = np.argmax(sizes) + 1

        markers[ markers != vessels_label] = 0
  
        show_images([markers], titles = ["vessels"])

        # mask out the markers
        mask [markers != 0] = 0
        self._mask = mask
        return markers
