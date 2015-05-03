import os, sys
import skimage
from skimage import io, util
import numpy as np

def cbalance(img, mask, s1, s2):
    h, bins = np.histogram(img,bins=256,range=(0,255))
    h[0] -= np.sum(1 - mask) # compensate for the borders
    histo = np.cumsum(h)    
    N = img.shape[0] * img.shape[1]    
    
    vmin = np.argmin(histo <= N * s1 / 100)
    vmax = np.argmax(histo > N * (1. - s2 / 100))
    vmax = 255 if vmax== 0 else vmax

    scale = 255. / (vmax - vmin)

    img = np.maximum(img, vmin) # saturate values smaller than the minimal
    img = np.minimum(img, vmax) # saturater values larger than vmax
    img = (img - vmin) * scale
    img = np.array(img, dtype=np.uint8)
    
    return img

def colorbalance(img, mask, s1, s2):
    ared = cbalance(np.array(img[:,:,0]), mask, s1, s2)
    agreen = cbalance(np.array(img[:,:,1]), mask, s1, s2)
    ablue = cbalance(np.array(img[:,:,2]), mask, s1, s2)
    
    return np.dstack((ared, agreen, ablue ))