import numpy as np
import pandas as pd
#from dark_bright_detector import DarkBrightDetector
from kobra.dr import Labels
from kobra import TrainFiles
from kobra.tr_utils import prep_out_path, time_now_str
import os
from os import path
import mahotas as mh
import mahotas.labeled as mhl
import cv2
import time

preprocessed = '/kaggle/retina/train/labelled'
masks = '/kaggle/retina/train/masks'
orig = '/kaggle/retina/train/sample/split'
output = '/kaggle/retina/train/sample/features'

n_bins = 100

prep_out_path(output)

for i in range(0, 5):
    prefix = str(i)

    print "Starting extraction @ ", time_now_str()
    files = os.listdir(path.join(preprocessed, prefix))
    
    # intermediate output will be stored here
    # we will save all the files first then join them into one csv file
    df = pd.DataFrame(columns = range(n_bins * 2 + 1))
    j = 0

    for f in files:
        start = time.time()
        
        im_file = path.join(prefix, f)

        extractor = DarkBrightDetector(preprocessed, orig, im_file, masks, is_debug = False)
        labels = extractor.find_bright_regions()

        drusen = extractor.get_predicted_region(Labels.Drusen)
        blood = extractor.get_predicted_region(Labels.Haemorage)

        Bc = np.ones((5, 5))
        labels_drusen, n_drusen = mh.label(drusen, Bc)
        labels_blood, n_blood = mh.label(blood, Bc)

        area = float(cv2.countNonZero(extractor.mask))

        outp = np.array([], dtype = np.int)

        # sizes excluding background
        sizes_drusen = mhl.labeled_size(labels_drusen)[1:] / area
        sizes_blood = mhl.labeled_size(labels_blood)[1:] / area

        hist_druzen, _ = np.histogram(sizes_drusen, n_bins, (0, 1e-3))
        hist_blood, _ = np.histogram(sizes_blood, n_bins, (0, 1e-3))

        outp = np.r_[outp, hist_druzen]
        outp = np.r_[outp, hist_blood]
        outp = np.r_[outp, i]
        df.loc[j] = outp
        j += 1
        print "Extracted: {0}, took {1:02.2f} sec ".format(im_file, time.time() - start)
      
    # write out the csv
    df.to_csv(path.join(output, prefix + ".txt"), index = False, header=False)    
    print "Extracted: ", prefix, "@", time_now_str()
