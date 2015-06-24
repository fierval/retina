import numpy as np
import pandas as pd
import cv2
from os import path
import os
import shutil
from kobra.imaging import *
import pywt
import mahotas as mh
from image_processor import ImageProcessor
from image_reader import ImageReader

root = '/kaggle/retina/train/sample/split'
im_file = '3/27224_right.jpeg'
masks_dir = '/kaggle/retina/train/masks'


class ExtractBloodVessels(ImageProcessor):
    '''
    Loosely based on: http://ictactjournals.in/paper/IJSCPaper_1_563to575.pdf
    and http://www4.comp.polyu.edu.hk/~cslzhang/paper/CBM-MF.pdf
    '''
    def __init__(self, root, im_file, masks_dir, is_debug = True):
        ImageProcessor.__init__(self, root, im_file, masks_dir)

        # keep green channel
        self._image = self.image[:, :, 1].copy()
        self._norm_const = 2.45
        self._is_debug = is_debug

    def detect_vessels(self):
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
        im_haar = remove_light_reflex(im_haar)
        
        mask = ImageReader.rescale_mask(im_haar, self.mask)
    
        # compute the image mean to "invert" the image
        # for the gausssian matched filter
        mean_thresh, _, _, _ = cv2.mean(im_haar, mask)
        im_haar = mean_thresh - im_haar

        # Gaussian Matched and FDOG filter responses
        K_MF = gaussian_matched_filter_kernel(31, 5)
        K_FDOG = fdog_filter_kernel(31, 5)
        kernels_mf = createMatchedFilterBank(K_MF, 12)
        kernels_fdog = createMatchedFilterBank(K_FDOG, 12)
        im_matched_mf = applyFilters(im_haar, kernels_mf)
        im_matched_fdog = applyFilters(im_haar, kernels_fdog)

        # normalize local mean of FDOG response
        local_mean_fdog = cv2.blur(im_matched_fdog, (11, 11))
        local_mean_fdog = cv2.normalize(local_mean_fdog, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
        
        # set the threshold matrix
        mean_thresh, _, _, _ = cv2.mean(im_matched_mf, mask)
        ref_thresh = mean_thresh * self._norm_const
        ref_thresh = ref_thresh * (1 + local_mean_fdog)

        # show the results
        if self._is_debug:
            show_images([self._reader.image])
            show_images([im_gray, im_haar], titles = ["gray", "haar"])
            show_images([im_matched_mf, im_matched_fdog], titles = ["mf", "fdog"])
        

        im_vessels = im_matched_mf.copy()
        im_vessels [im_vessels < ref_thresh] = 0
        im_vessels [mask == 0] = 0
        im_vessels [im_vessels != 0] = 1
        im_vessels = im_vessels.astype('uint8')

        if self._is_debug:
            show_images([im_vessels], titles = ["vessels"])
            self._im_norm = mh.stretch(im_matched_mf)

        return ImageReader.rescale_mask(self.image, im_vessels)