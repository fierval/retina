import numpy as np
import pandas as pd
from kobra.dr import Labels
from kobra import TrainFiles
from kobra.tr_utils import prep_out_path, time_now_str
import os
from os import path
from sklearn.cross_validation import train_test_split
import shutil
import mahotas as mh
import mahotas.labeled as mhl
import cv2
from kobra.dr import ImageReader
import time

proc_path = '/kaggle/retina/reduced/train'
labels_file =  '/kaggle/retina/trainLabels.csv'
root = '/kaggle/retina/reduced/sample'

masks_dir = '/kaggle/retina/train/prepmasks'
features_path = '/kaggle/retina/reduced/features/train'
prefix = 'features'


def get_predicted_region(im, marker):
    res = im.copy()
    res[res != marker] = 0
    return res

def get_areal_features(root, features_path, masks_dir, n_bins = 100):
    prep_out_path(features_path)
    files = os.listdir(root)

    df = pd.DataFrame(columns = range(n_bins * 2) + ['name', 'level'])
    names = pd.read_csv(labels_file)
    print "Starting extraction: ", time_now_str()

    for j, f in enumerate(files):
        label = names.loc[names['image'] == path.splitext(f)[0]]
        start = time.time()
        imr = ImageReader(root, f, masks_dir, gray_scale = True)

        drusen = get_predicted_region(imr.image, Labels.Drusen)
        blood = get_predicted_region(imr.image, Labels.Haemorage)

        Bc = np.ones((5, 5))
        labels_drusen, n_drusen = mh.label(drusen, Bc)
        labels_blood, n_blood = mh.label(blood, Bc)

        area = float(cv2.countNonZero(imr.mask))

        outp = np.array([], dtype = np.int)

        # sizes excluding background
        sizes_drusen = mhl.labeled_size(labels_drusen)[1:] / area
        sizes_blood = mhl.labeled_size(labels_blood)[1:] / area

        hist_druzen, _ = np.histogram(sizes_drusen, n_bins, (0, 1e-3))
        hist_blood, _ = np.histogram(sizes_blood, n_bins, (0, 1e-3))


        outp = np.r_[outp, hist_druzen]
        outp = np.r_[outp, hist_blood]
        outp = np.r_[outp, label.values[0]]
        df.loc[j] = outp
        print "Extracted: {0}, took {1:02.2f} sec ".format(f, time.time() - start)
      
    # write out the csv
    df.to_csv(path.join(features_path, prefix + ".csv"), index = False, header=True)    
    print "Extracted: ", prefix, "@", time_now_str()
