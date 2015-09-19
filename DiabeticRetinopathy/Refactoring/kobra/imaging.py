from matplotlib import pyplot as plt
from matplotlib import cm
from os import path
import numpy as np
import cv2
import pandas as pd
from math import exp, pi, sqrt
import mahotas as mh
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
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()
    plt.close()
    
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

def remove_light_reflex(im, ksize = 5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)

    # an un-natural way to set elements of the array
    # to their x coordinate
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    @vectorize(['float32(float32)'], target='cpu')
    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    @vectorize(['float32(float32)'], target='cpu')
    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
       kernel = k_fun_derivative(arr)

    # return the correlation kernel for filter2D
    return cv2.flip(kernel, -1) 

def fdog_filter_kernel(L, sigma, t = 3):
    '''
    K = - (x/(sqrt(2 * pi) * sigma ^3)) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, False)

def gaussian_matched_filter_kernel(L, sigma, t = 3):
    '''
    K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, True)

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

def saturate (v):
    return np.array(map(lambda a: min(max(round(a), 0), 255), v))

def calc_hist(images, masks):
    channels = map(lambda i: cv2.split(i), images)
    imMask = zip(channels, masks)
    nonZeros = map(lambda m: cv2.countNonZero(m), masks)
    
    # grab three histograms - one for each channel
    histPerChannel = map(lambda (c, mask): \
                         [cv2.calcHist([cimage], [0], mask,  [256], np.array([0, 255])) for cimage in c], imMask)
    # compute the cdf's. 
    # they are normalized & saturated: values over 255 are cut off.
    cdfPerChannel = map(lambda (hChan, nz): \
                        [saturate(np.cumsum(h) * 255.0 / nz) for h in hChan], \
                        zip(histPerChannel, nonZeros))
    
    return np.array(cdfPerChannel)

# compute color map based on minimal distances beteen cdf values of ref and input images    
def getMin (ref, img):
    l = [np.argmin(np.abs(ref - i)) for i in img]
    return np.array(l)

# compute and apply color map on all channels of the image
def map_image(image, refHist, imageHist):
    # each of the arguments contains histograms over 3 channels
    mp = np.array([getMin(r, i) for (r, i) in zip(refHist, imageHist)])

    channels = np.array(cv2.split(image))
    mappedChannels = np.array([mp[i,channels[i]] for i in range(0, 3)])
    
    return cv2.merge(mappedChannels).astype(np.uint8)

# compute the histograms on all three channels for all images
def histogram_specification(ref, images, masks):
    '''
    ref - reference image
    images - a set of images to have color transferred via histogram specification
    masks - masks to apply
    '''
    cdfs = calc_hist(images, masks)
    mapped = [map_image(images[i], ref[0], cdfs[i, :, :]) for i in range(len(images))]
    return mapped

def max_labelled_region(labels, Bc = None):
    '''
    Labelled region of maximal area
    '''
    return np.argmax(mh.labeled.labeled_size(labels)[1:]) + 1

def saturate (v):
    return map(lambda a: min(max(round(a), 0), 255), v)


def plot_hist(hst, color):
    fig = plt.figure()
    plt.bar(np.arange(256), hst, width=2, color=color, edgecolor='none')
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * 2)
    plt.show()
