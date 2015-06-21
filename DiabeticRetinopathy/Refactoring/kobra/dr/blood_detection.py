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
    http://www4.comp.polyu.edu.hk/~cslzhang/paper/CBM-MF.pdf
    '''
    def __init__(self, root, im_file, masks_dir):
        ImageProcessor.__init__(self, root, im_file, masks_dir)

        # keep green channel
        self._image = self.image[:, :, 1].copy()

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
        im_haar = remove_light_reflex(im_haar)

        # Matched and FDOG filter responses
        K_MF = matched_filter_kernel(15, 3, 5)
        K_FDOG = fdog_filter_kernel(15, 3, 5)
        kernels_mf = createMatchedFilterBank(K_MF, 12)
        kernels_fdog = createMatchedFilterBank(K_FDOG, 12)
        im_matched_mf = applyFilters(im_haar, kernels_mf)
        im_matched_fdog = applyFilters(im_haar, kernels_fdog)

        # show the results
        show_images([self._reader.image])
        show_images([im_gray, im_haar], titles = ["gray", "haar"])
        show_images([im_matched_mf, im_matched_fdog], titles = ["mf", "fdog"])

        return mh.stretch(im_matched_fdog.astype(np.int))

    def extract_blood_vessels_mask(self, im_norm):
        '''
        Returns blood vessels and 'adjacent' 
        pixel effects to be masked out.
        '''

        # Haar changes image size. We need to change the mask size as well
        mask = ImageReader.rescale_mask(im_norm, self.mask)
        thresh, _, _, _ = cv2.mean(im_norm, mask)

        # computes vessel regions with mahotas distance function
        Bc = np.ones((3, 3))
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