import numpy as np
import pandas as pd
import cv2
from os import path
import os
import shutil
from kobra.imaging import show_images
import pywt
from numbapro import vectorize

root = '/kaggle/retina/train/sample/split'
im_file = '3/27224_right.jpeg'
masks_dir = '/kaggle/retina/train/masks'

def matched_filter_kernel(L, sigma):
    '''
    K = -exp(-x^2/2sigma^2), |y| <= L/2
    '''
    dim = int(L/2)
    arr = np.zeros((dim, dim), 'f')

    # an un-natural way to set elements of the array
    # to their x coordinate
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[0]
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma

    @vectorize(['float32(float32)'], target='cpu')
    def k_fun(x):
        return -exp(-x * x / two_sigma_sq)

    return k_fun(arr)


def preprocess(root, im_file):
    im_path = path.join(root, im_file) 

    im = cv2.imread(im_path)

    im = cv2.resize(im, scale)
    mask_file = path.join(masks_dir, path.splitext(path.split(im_file)[1])[0] + ".png")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    scale = im.shape[1], im.shape[0]
    mask = cv2.resize(mask, scale)

    im[mask == 0] = 0

    # grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(4, (5, 5))
    im_clahe = clahe.apply(im_gray)

    #Haar transform
    wp = pywt.WaveletPacket2D(im_clahe, wavelet = 'haar', maxlevel = 1, mode = 'sym')
    wp.decompose()
    im_haar = wp['a'].data

    