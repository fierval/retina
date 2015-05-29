from matplotlib import pyplot as plt
from os import path
import numpy as np
import cv2
import pandas as pd


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

def find_eye(image, thresh = 4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge finder
    edges = np.array([])
    edges = cv2.Canny(gray, thresh, thresh * 3, edges)

    # Find contours
    # second output is hierarchy - we are not interested in it.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Now let's get only what we need out of it
    hull_contours = cv2.convexHull(np.vstack(np.array(contours)))
    hull = np.vstack(hull_contours)
    
    def createMask((rows, cols), hull):
        # black image
        mask = np.zeros((rows, cols), dtype=np.uint8)
        # blit our contours onto it in white color
        cv2.drawContours(mask, [hull], 0, 255, -1)
        return mask

    mask = createMask(image.shape[0:2], hull)
    
    # returning the hull to illustrate a few issues below
    return mask, hull

def saturate (v):
    return np.array(map(lambda a: min(max(round(a), 0), 255), v))

def get_masks(images, thresh=4):
    return map(lambda i: find_eye(i, thresh)[0], images)

def maskToAreaRatio(images):
    return map(lambda im: float(cv2.countNonZero(im)) / (im.shape[0] * im.shape[1]), images)
   
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
        cdfs = calc_hist(images, masks)
        mapped = [map_image(images[i], ref[0], cdfs[i, :, :]) for i in range(len(images))]
        return mapped

def mask_background(image, mask):
    channels = np.array(cv2.split(image))
    
    # now mask off the backgrownd
    for i in range(0, 3):
        channels[i, mask == 0] = 0
    return cv2.merge(channels).astype(np.uint8)

thresh = 4
img_path = "/kaggle/retina/train/sample"
# this is the image into the colors of which we want to map
ref_image_name = "6535_left.jpeg"
# images picked to illustrate different problems arising during algorithm application
image_names = ["16_left.jpeg", "10130_right.jpeg", "21118_left.jpeg"]

def pre_process(img_path, image_names, ref_image_name, thresh):
    image_paths = map(lambda t: path.join(img_path, t), image_names)
    images = np.array(map(lambda p: cv2.imread(p), image_paths))
    image_titles = map(lambda i: path.splitext(i)[0], image_names)

    images = np.array(map(lambda im: pyr_blurr(im), images))

    # get the masks
    masks = np.array(get_masks(images, thresh))
    areas = np.array(maskToAreaRatio(masks))

    imageMaskLarge = images[areas > 0.41]
    imageMaskSmall = images[areas <= 0.41]

    masksLarge = masks[areas > 0.41].tolist()
    masksSmall = get_masks(imageMaskSmall.tolist(), 1)

    images = np.concatenate((imageMaskLarge, imageMaskSmall))
    # when trying to convert a list of one element to array, the result is not
    # an array of array, but one multi-dimensional array
    masks = masksLarge + masksSmall
    
    ref_image = cv2.imread(path.join(img_path, ref_image_name))
    ref_image = pyr_blurr(ref_image)

    refMask = get_masks([ref_image])
    refHist = calc_hist([ref_image], refMask)

    histSpec = histogram_specification(refHist, images, masks)

    # mask off the background
    maskedBg = [mask_background(h, m) for (h, m) in zip(histSpec, masks)]
    show_images(maskedBg, titles = image_titles, scale = 0.9)
