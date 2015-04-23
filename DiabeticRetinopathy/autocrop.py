import os
import skimage
from skimage import io, util
import numpy as np

def autocrop(img):
    s = np.sum(img, axis=2)
    cols = np.sum(s, axis=0) 
    rows = np.sum(s, axis=1)
    left_border = np.argmax(cols[0:len(cols)/2] > 10000)
    right_border = len(cols)/2 - np.argmax(cols[len(cols)/2:len(cols)-1] < 10000)
    upper_border = np.argmax(rows[0:len(rows)/2] > 10000)
    lower_border = len(rows)/2 - np.argmax(rows[len(rows)/2:len(rows)-1] < 10000)
    
    return skimage.util.crop(img, ((upper_border, lower_border),(left_border, right_border),  (0,0)))