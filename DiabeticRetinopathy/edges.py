from skimage import io
from os import path
from matplotlib import pyplot as plt
from skimage.filter import canny
import numpy as np

root = path.normpath('h:/kaggle/retina/train')
an_eye = path.join(root, '84_left.jpeg')
eye = io.imread(an_eye)

# get just one channel
red_eye = eye[:, :, 0]

edge = canny(red_eye, sigma=0.1)

#crop using numpy
row_dim, col_dim = np.nonzero(edge)

x_top, y_left = row_dim[0], np.min(col_dim)
x_bottom, y_right = row_dim[-1], np.max(col_dim)

cropped = eye[x_top : x_bottom, y_left : y_right, :]

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(eye)
ax2.imshow(cropped)
ax3.imshow(edge)
