import numpy as np
import pandas as pd
import cv2
from os import path
import os
import shutil
from kobra.imaging import show_images
import pywt
from numbapro import vectorize
import mahotas as mh

root = '/kaggle/retina/train/sample/split'
im_file = '3/27224_right.jpeg'
masks_dir = '/kaggle/retina/train/masks'

def remove_light_reflex(im):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

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

    rotate = 180 / n
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

def get_mask(im, im_file, masks_dir):
    mask_file = path.join(masks_dir, path.splitext(path.split(im_file)[1])[0] + ".png")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    scale = im.shape[1], im.shape[0]
    mask = cv2.resize(mask, scale)
    return mask

def preprocess(root, im_file, masks_dir):
    im_path = path.join(root, im_file) 

    im = cv2.imread(im_path)

    mask_file = path.join(masks_dir, path.splitext(path.split(im_file)[1])[0] + ".png")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    scale = im.shape[1], im.shape[0]
    mask = get_mask(im, im_file, masks_dir)

    im[mask == 0] = 0

    # green channel & remove light reflex
    im_gray = remove_light_reflex(im[:, :, 1])

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
    show_images([im])
    show_images([im_gray, im_haar], titles = ["gray", "haar"])
    show_images([im_norm], titles = ["filtered"])

    return im_norm

def extract_blood_vessels(im_norm, im_file, masks_dir):
    '''
    Extracts blood vessels. 

    im_norm - output of preprocess()
    '''
    mask = get_mask(im_norm, im_file, masks_dir)
    thresh, _, _, _ = cv2.mean(im_norm, mask)

    # computes vessel regions with mahotas distance function
    Bc = np.ones((3, 3))
    threshed = (im_norm > thresh)
    distances = mh.stretch(mh.distance(threshed))
    _, im = cv2.threshold(distances, 0, 255, cv2.THRESH_BINARY)
    
    # erode/dilate 
    im = remove_light_reflex(im)
    im = remove_light_reflex(im)
    
    # label and remove the region of the largest size
    markers, n_markers = mh.label(im)

    # sizes without the background region
    sizes = mh.labeled.labeled_size(markers)[1:]

    # vessels region is the one with the largest area
    # since we have removed background region "0", add 1
    vessels_label = np.argmax(sizes) + 1

    markers[ markers != vessels_label] = 0
  
    show_images([markers], titles = ["vessels"])

    return markers