from skimage import io
from os import path
from matplotlib import pyplot as plt
from skimage.feature import canny
import numpy as np
import cv2
from skimage.color.colorconv import rgb2gray
from autocrop import crop_manual

in_path = path.normpath('c:/kaggle/retina/train/raw')
out_path = path.normpath('c:/kaggle/retina/train/cropped')

def do_cropping(eye, edge):
    row_dim, col_dim = np.nonzero(edge)

    x_top, y_left = row_dim[0], np.min(col_dim)
    x_bottom, y_right = row_dim[-1], np.max(col_dim)

    return eye[x_top : x_bottom, y_left : y_right, :]

def crop_cv(eye):
    red_eye = eye[:, :, 0]

    # compute the Otsu th
    thresh_value, _ = cv2.threshold(red_eye, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    upper, lower = thresh_value, thresh_value / 2
    edges = cv2.Canny(eye, upper, lower)
    return do_cropping(eye, edges)

def crop_sk(eye):
    red_eye = rgb2gray(eye)

    edges = canny(red_eye, sigma=3)
    return do_cropping(eye, edges)

def test():
    an_eye = path.join(root, '20677_left.jpeg')
    eye = io.imread(an_eye)
    #eye_cv = cv2.imread(an_eye, cv2.IMREAD_GRAYSCALE)


    # get just one channel
    cropped = crop_sk(eye)
    cropped1 = crop_cv(eye)
    cropped2 = crop_manual(eye)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(eye)
    ax2.imshow(cropped)
    ax3.imshow(cropped1)
    ax4.imshow(cropped2)
    plt.show()