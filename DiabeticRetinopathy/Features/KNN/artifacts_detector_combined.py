import numpy as np
import pandas as pd

import cv2
from knn_detectors import HaemDetect, DrusenDetect
from kobra.imaging import show_images

root_drusen = '/kaggle/retina/train/labelled'
root_haem = '/kaggle/retina/train/labelled_1'

annotations = 'annotations.txt'
masks_dir = '/kaggle/retina/train/masks'
im_file = '4/16_left.jpeg'
orig_path = '/kaggle/retina/train/sample'

def detect(root_drusen, root_haem, annotations, orig_path, masks_dir, im_file):
    drusen = DrusenDetect(root_drusen, annotations, masks_dir, orig_path, n_neighbors = 3)
    haem = HaemDetect(root_haem, annotations, masks_dir, orig_path, n_neighbors = 5)

    haem.analyze_image(im_file)
