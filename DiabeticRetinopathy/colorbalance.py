import os, sys
import skimage
from skimage import io, util
import numpy as np

def cbalance(img, s1, s2):
    
    img_f = img.flatten()
    h, bins = np.histogram(img_f,bins=256,range=(0,255))
    histo = np.cumsum(h)    
    N = len(img_f)
    
    vmin = 0
    while histo[vmin + 1] <= N * s1 / 100:
        vmin = vmin + 1
    vmax = 255 - 1
    while histo[vmax - 1] > N * (1 - s2 / 100):
        vmax = vmax - 1
    if vmax < 255 - 1:
        vmax = vmax + 1
        
    scale = 255. / (vmax - vmin)

    img_f = np.maximum(img_f, vmin) # saturate values smaller than the minimal
    img_f = np.minimum(img_f, vmax) # saturater values larger than vmax
    img_f = (img_f - vmin) * scale
    img_f = np.array(img_f, dtype=int)
    return np.reshape(img_f, img.shape)

def colorbalance(img,s1, s2):
    img1 = np.zeros(img.shape)

    img1[:,:,0] = cbalance(img[:,:,0], 1, 1)
    img1[:,:,1] = cbalance(img[:,:,1], 1, 1)
    img1[:,:,2] = cbalance(img[:,:,2], 1, 1)
    
    return img1