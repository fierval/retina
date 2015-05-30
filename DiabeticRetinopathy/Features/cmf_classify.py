import cv2
import pandas as pd
import numpy as np
import skfuzzy.cluster as cmf
from os import path
import os
import shutil

class CmfClassify (object):
    def __init__(self, root, annotations):
        self._root = root
        self._annotations = annotations
        self._pd_annotations = pd.read_csv(self._annotations, sep= ' ', header = None)

        # files from which annotations were extracted
        self._files = self._pd_annotations[0].as_matrix()

        # number of rectangles in each file
        self._n_objects = self._pd_annotations[1][0]

        rect_frame = self._pd_annotations.ix[:, 2:].as_matrix()
        rows = rect_frame.shape[0]
        rect_frame = np.split(rect_frame, rows)
        self._rects = np.array([])
        
        for row in rect_frame:
            rects = np.array([])
            for i in range(0, self._n_objects):
                rect = (0, 0, 0, 0)
                for j in range(0, 4):
                    rect[j] = row[i * 4 + j]
                rects = np.r_["1", rects, np.array([rect])]
            np.vstack((self._rects, rects))                            