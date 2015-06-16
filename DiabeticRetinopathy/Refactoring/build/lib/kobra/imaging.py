from matplotlib import pyplot as plt
from os import path
import numpy as np
import cv2
import pandas as pd
from math import exp
from numbapro import vectorize


def show_images(images,titles=None, scale=1.3):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.imshow(image)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()
    
# Pyramid Down & blurr
# Easy-peesy
def pyr_blurr(image):
    return cv2.GaussianBlur(cv2.pyrDown(image), (7, 7), 30.)

def median_blurr(image, size = 7):
    return cv2.medianBlur(image, size)

def display_contours(image, contours, color = (255, 0, 0), thickness = -1, title = None):
    imShow = image.copy()
    for i in range(0, len(contours)):
        cv2.drawContours(imShow, contours, i, color, thickness)
    show_images([imShow], scale=0.7, titles=title)

def salt_and_peper(im, fraction = 0.01):

    assert (0 < fraction <= 1.), "Fraction must be in (0, 1]"

    sp = np.zeros(im.shape)
    percent = round(fraction * 100 / 2.)

    cv2.randu(sp, 0, 100)

    # quarter salt quarter pepper
    im_sp = im.copy()
    im_sp [sp < percent] = 0
    im_sp [sp > 100 - percent] = 255
    return im_sp

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
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)

    return kernels

def applyFilters(im, kernels):
    '''
    Given a filter bank, apply them and record maximum response
    '''
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

def gabor_filters(ksize, sigma = 4.0, lmbda = 10.0, n = 16):
    '''
    Create a bank of Gabor filters spanning 180 degrees
    '''
    filters = []
    for theta in np.arange(0, np.pi, np.pi / n):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lmbda, 0.5, 0, ktype=cv2.CV_64F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
