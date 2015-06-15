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

def createMatchedFilterBank(K, n = 12):
    '''
    Given a kernel, create matched filter bank
    '''

    rotate = 360 / n
    center = (K.shape[0] / 2, K.shape[1] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, K.shape)
        kernels.append(k)

    return kernels

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

def gabor_filters(ksize, sigma = 4.0, lmbda = 10.0, n = 16):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / n):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lmbda, 0.5, 0, ktype=cv2.CV_64F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def preprocess(root, im_file, masks_dir):
    im_path = path.join(root, im_file) 

    im = cv2.imread(im_path)

    mask_file = path.join(masks_dir, path.splitext(path.split(im_file)[1])[0] + ".png")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    scale = im.shape[1], im.shape[0]
    mask = cv2.resize(mask, scale)

    #im[mask == 0] = 0

    # grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(4, (5, 5))
    im_clahe = clahe.apply(im_gray)

    # Haar transform
    wp = pywt.WaveletPacket2D(im_clahe, wavelet = 'haar', maxlevel = 1, mode = 'sym')
    wp.decompose()
    im_haar = wp['a'].data

    # Matched Filter Response
    K = matched_filter_kernel(31, 5)
    kernels = createMatchedFilterBank(K, 12)
    im_matched = applyFilters(im_haar, kernels)
    kernels = gabor_filters(13, n = 12, sigma = 4.5)
    im_matched = applyFilters(im_matched, kernels)
    im_norm = cv2.normalize(im_matched, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
#    mask = cv2.resize(mask, (im_norm.shape[1], im_norm.shape[0]))
#    im_norm [mask == 0] = 0

    return im_norm