from matplotlib import pyplot as plt
from os import path
import numpy as np
import cv2
import pandas as pd
from kobra.imaging import *
from kobra.dr.retina import createMask, find_eye, display_contours

def get_masks(images, thresh=4):
    return map(lambda i: find_eye(i, thresh)[0], images)

def maskToAreaRatio(images):
    return map(lambda im: float(cv2.countNonZero(im)) / (im.shape[0] * im.shape[1]), images)

def mask_background(image, mask):
    channels = np.array(cv2.split(image))
    
    # now mask off the backgrownd
    for i in range(0, 3):
        channels[i, mask == 0] = 0
    return cv2.merge(channels).astype(np.uint8)

thresh = 4
img_path = "/kaggle/retina/train/sample"
# this is the image into the colors of which we want to map
ref_image_name = "16_left.jpeg"
# images picked to illustrate different problems arising during algorithm application
#image_names = ["16_left.jpeg", "10130_right.jpeg", "21118_left.jpeg"]
image_names = ["6535_left.jpeg", "10130_right.jpeg", "21118_left.jpeg"]

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
    # an array of arrays, but one multi-dimensional array
    masks = masksLarge + masksSmall
    
    ref_image = cv2.imread(path.join(img_path, ref_image_name))
    ref_image = pyr_blurr(ref_image)

    refMask = get_masks([ref_image])
    refHist = calc_hist([ref_image], refMask)

    histSpec = histogram_specification(refHist, images, masks)

    # mask off the background
    maskedBg = [mask_background(h, m) for (h, m) in zip(histSpec, masks)]
    show_images(maskedBg, titles = image_titles, scale = 0.9)
