import numpy as np
import pandas as pd
import os
from os import path
from skimage import io
from skimage.io import ImageCollection
from skimage.exposure.exposure import equalize_hist
from skimage.color.colorconv import rgb2gray
from skimage.feature.blob import blob_doh
from skimage.feature.corner import corner_peaks, corner_harris

root_path = "/kaggle/retina"

# train/test directories
train_path = path.join(root_path, 'train')
sample_train = path.join(train_path, 'sample')

# in CSV representation
labels_file = path.join(root_path, "trainLabels.csv")
labels = pd.read_csv(labels_file, header=0)

def get_image_name(file_name):
    return path.splitext(path.split(file_name)[1])[0]

def process_single_image(file_name):
    image = io.imread(file_name)
    image_gray = equalize_hist(rgb2gray(image))

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
    corners = corner_peaks(corner_harris(image_gray), min_distance=2)
    image_name = get_image_name(file_name)
    level = labels[labels['image'] == image_name]['level'].iloc[0]

    return np.array([blobs_doh.shape[0], corners.shape[0], int(level != 0)])