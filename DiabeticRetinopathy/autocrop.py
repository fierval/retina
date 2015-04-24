import numpy as np
from skimage.util import crop

def crop_manual(img):
    threshold = 20000
    s = np.sum(img, axis=2)
    cols = np.sum(s, axis=0) > threshold  
    rows = np.sum(s, axis=1) > threshold
    
    left_border = np.argmax(cols[0:len(cols)/2])
    right_border = np.argmax(cols[len(cols)-1:len(cols)/2:-1])
    upper_border = np.argmax(rows[0:len(rows)/2])
    lower_border = np.argmax(rows[len(rows)-1:len(rows)/2:-1])
  
    return crop(img, ((upper_border, lower_border),(left_border, right_border),  (0,0)))