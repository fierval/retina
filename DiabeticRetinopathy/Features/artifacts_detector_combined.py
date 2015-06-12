import numpy as np
import pandas as pd

import cv2
from regions_detect_knn import KNeighborsRegions


root_drusen = '/kaggle/retina/train/labelled'
root_haem = '/kaggle/retina/train/labelled_1'

annotations = 'annotations.txt'
masks_dir = '/kaggle/retina/train/masks'
im_file = '4/16_left.jpeg'
orig_path = '/kaggle/retina/train/sample'

def detect(root_drusen, root_haem, annotations, orig_path, masks_dir, im_file):
    drusen = KNeighborsRegions(root_drusen, annotations, masks_dir, orig_path, n_neighbors = 3)
    haem = KNeighborsRegions(root_haem, annotations, masks_dir, orig_path, n_neighbors = 5)

    haem.analyze_image(im_file)
