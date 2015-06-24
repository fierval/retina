import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import path
import os
import shutil
from kobra.tr_utils import append_to_arr
from kobra.imaging import show_images, pyr_blurr
from kobra import enum
from kobra.dr.retina import find_eye
from kobra.dr import ImageProcessor

# annotation labels
# Background == Texture. Masked == irrelevant for the prediction matrix
# Used to mask regions out of the matrix returned by KNN
Labels = enum(Drusen = 7, Background = 1, BloodVessel = 2, 
              CameraHue = 3, Haemorage = 4, OD = 5, Masked = 6)

def merge_annotations(a1_file, a2_file, out_file = None):
    '''
    Helper function to merge two annotation files
    '''
    if out_file == None: out_file = a1_file
    a1 = pd.read_csv(a1_file, sep = ' ', header = None)
    a2 = pd.read_csv(a2_file, sep = ' ', header = None)

    #number of new objects
    new_obj = a2[1][0]
    del a2[1]

    out = pd.merge(a1, a2, on = 0)
    out[1] += new_obj

    out.to_csv(out_file, sep = ' ', header = None, index = False)
     
class KNeighborsRegions (ImageProcessor):
    def __init__(self, root, im_file, annotations, masks_dir, n_neighbors = 3):
        '''
        root - root directory for images
        masks_dir - where masks are
        annotations - path to the annotations file
        orig_path - path to the original (unprocessed) files

        Images must be pre-processed by FeatureExtractionCpp
        '''

        ImageProcessor.__init__(self, root, im_file, masks_dir)
        self._root = root

        self._annotations = path.join(root, annotations)
        self._n_neighbors = n_neighbors

        assert(path.exists(self._annotations)), "Annotations file does not exist: " + self._annotations

        self._pd_annotations = pd.read_csv(self._annotations, sep= ' ', header = None)

        self._rects = np.array([])
        self._avg_pixels = np.array([])
        self._labels = None
        
        self._process_annotations()        

    def _process_annotations(self):
        # files from which annotations were extracted
        self._files = self._pd_annotations[0].as_matrix()
        self._n_files = len(self._files)

        # number of rectangles in each file
        self._n_objects = self._pd_annotations[1][0]
        rect_frame = self._pd_annotations.ix[:, 2:].as_matrix()
        rows = rect_frame.shape[0]
        rect_frame = np.vsplit(rect_frame, rows)

        # convert number-by-number columns into rectangles
        for row in rect_frame:
            row = row.reshape(-1)
            rects = np.array([], dtype=('i4, i4, i4, i4'))
            for i in range(0, self._n_objects):
                idx = i * 4
                # images are loaded by OpenCV with rows and columns mapped to y and x coordinates
                # rectangles are stored by annoations tool as x, y, dx, dy -> col, row, dcol, drow
                rect = row[idx + 1], row[idx], row[idx + 1] + row[idx + 3], row[idx] + row[idx + 2]
                if rects.size == 0:
                    rects = np.array(rect, dtype=('i4, i4, i4, i4'))
                else:
                    rects = np.append(rects, np.array(rect, dtype=('i4, i4, i4, i4')))
            
            self._rects = append_to_arr(self._rects, rects)                            

    def _get_initial_classes(self):
        images = map(lambda f: cv2.imread(path.join(self._root, f)), self._files)
        self._avg_pixels = np.array([], dtype=np.uint8)

        # extract parts from each image for all of our 6 categories
        for i in range(0, self._n_objects):
            rects = self._rects[:, i]
            
            # compute maximum rectangle
            rows = np.max(rects['f2'] - rects['f0'])
            cols = np.max(rects['f3'] - rects['f1'])

            # extract annotated rectangles
            im_rects = map(lambda (im, r): im[r[0]:r[2],r[1]:r[3],:], zip(images, rects))

            # resize all rectangles to the max size & average all the rectangles
            im_rects = np.array(map(lambda im: cv2.resize(im, (cols, rows)), im_rects), dtype=np.float)
            avgs = np.around(np.average(im_rects, axis = 0))

            # average the resulting rectangle to compute 
            mn = np.around(np.array(cv2.mean(avgs), dtype='float'))[:-1].astype('uint8')

            if(self._avg_pixels.size == 0):
                self._avg_pixels = mn
            else:
                self._avg_pixels = np.vstack((self._avg_pixels, mn))
        
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val

    def display_average_pixels(self):
        if self._avg_pixels == None:
            self._get_initial_classes()

        def stretch_image(i):
            pixel = self._avg_pixels[i]
            stretch = np.zeros((20, 20, 3), dtype='uint8')
            stretch[:, :] = pixel
            return stretch

        images = [stretch_image(i) for i in range(0, self._n_objects)]

        show_images(images)

    
    def display_artifact(self, prediction, artifact, color, title):
        im = self._image
        im_art = im.copy()

        im_art [prediction == artifact] = color

        show_images([im, im_art], ["original", title], scale = 0.8)

    def display_current(self, prediction, with_camera = False):
        pass
    
    def display_camera_artifact(self, prediction):
        self.display_artifact(prediction, Labels.CameraHue, (0, 0, 255), "camera")
    
    def display_bg(self, prediction):
        self.display_artifact(prediction, Labels.Background, (0, 0, 255), "background")

    def analyze_image(self):
        '''
        Load the image and analyze it with KNN

        im_file - pre-processed with histogram specification
        '''

        if self._avg_pixels.size == 0:
            self._get_initial_classes()
        
        
        im = self._image
        rows = im.shape[0]

        clf = KNeighborsClassifier(n_neighbors = self._n_neighbors)
        clf.fit(self._avg_pixels, self._labels)

        im_1d = im.reshape(-1, 3)

        # calculate prediction reshape into image
        prediction = clf.predict(im_1d)
        prediction = prediction.reshape(rows, -1)

        prediction [self._mask == 0] = Labels.Masked
        self.display_current(prediction)
        return prediction

    def _flood_fill(self, ctr, labels, newVal = 0, deltaLow = 0, deltaHigh = 0):

        # The mask needs to be 2 pixels larger than the image
        h, w = self._mask.shape[0] + 2, self._mask.shape[1] + 2
        mask = cv2.bitwise_not(cv2.resize(self._mask, (w, h)))

        # connectivity is 8 neighbors, fill mask with 255
        flags = 8 | ( 255 << 8 ) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(prediction, mask, ctr, newVal, deltaLow, deltaHigh, flags)

        self.display_current(labels)

        return labels