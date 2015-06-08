import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import path
import os
import shutil
from kobra.tr_utils import append_to_arr
from kobra.imaging import show_images, pyr_blurr
from od_detection import DetectOD

root = '/kaggle/retina/train/labelled'
annotations = 'annotations.txt'
masks_dir = '/kaggle/retina/train/masks'
im_file = '4/16_left.jpeg'
orig_path = '/kaggle/retina/train/sample'
orig_im_file = path.join(orig_path, '16_left.jpeg')

class KNeighborsRegions (object):
    def __init__(self, root, annotations, masks_dir):
        '''
        root - root directory for images
        annotations_images_dir - where annotations images are
        masks_dir - where masks are
        annotations - path to the annotations file

        Images must be pre-processed by FeatureExtractionCpp
        '''
        self._masks_dir = masks_dir
        self._root = root
        self._annotations = path.join(self._root, annotations)
        self._image = np.array([])

        assert(path.exists(self._annotations)), "Annotations file does not exist: " + self._annotations

        self._pd_annotations = pd.read_csv(self._annotations, sep= ' ', header = None)

        self._rects = np.array([])
        self._avg_pixels = np.array([])
        self._labels = [-1, -1, 1, 1, 2, 3, 4, 4, 5]
        
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
        '''
        Averages of the pixels values of all images:
        Annotations contain:
        position: meaning (label)
        0 - 1: drusen/exudates (0)
        2 - 3: background (1)
            4: vessels (2)
            5: haemorages (3)
            6: background (4) 
            7: bluish hue at the edges (5) 
        '''
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
        
        # append the background pixel
        self._avg_pixels = np.vstack((self._avg_pixels, np.array([0, 0, 0])))
            

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val

    @staticmethod
    def flip(rect):
        '''
        useful for converting between (pt1, pt2) and row x col representations of rectangles
        '''
        return rect[1], rect[0], rect[3], rect[2]
    
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

    
    def display_current(self, prediction, with_camera = False):
        # display what we have found
        im = self._image
        mask = self._mask
        im_drusen = im.copy()
        im_bg = im.copy()

        im_drusen [prediction == -1] = [255, 0, 0]
        if with_camera:
            im_drusen [prediction == 4] = [255, 0, 0]
        im_bg [prediction == 2] = [0, 255, 0]
        im_bg [mask == 0] = 0

        show_images([im, im_drusen, im_bg], ["original", "HE/CWS", "HM/MA"], scale = 0.8)
    
    def display_camera_artifact(self, prediction):
        im = self._image
        mask = self._mask
        im_camera = im.copy()

        im_camera [prediction == 4] = [255, 0, 0]

        show_images([im, im_camera], ["original", "camera"], scale = 0.8)

    def analyze_image(self, im_file):
        '''
        Load the image and analyze it with KNN

        im_file - pre-processed with histogram specification
        '''

        im_file = path.join(self._root, im_file)
        assert (path.exists(im_file)), "Image file does not exist"

        if self._avg_pixels.size == 0:
            self._get_initial_classes()

        im_file_name = path.splitext(path.split(im_file)[1])[0]

        mask_file = path.join(self._masks_dir, im_file_name + ".png")
        assert (path.exists(mask_file)), "Mask file does nto exist"

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        mask [ mask != 0 ] = 255
        rows = mask.shape[0]

        im = cv2.imread(im_file)

        assert (im.shape[0] == rows), "Rows don't match between mask and image"

        im [ mask == 0] = 0
        self._image = im
        self._mask = mask

        clf = KNeighborsClassifier(n_neighbors = 3)
        clf.fit(self._avg_pixels, self._labels)

        im_1d = im.reshape(-1, 3)

        # calculate prediction reshape into image
        prediction = clf.predict(im_1d)
        prediction = prediction.reshape(rows, -1)

        self.display_current(prediction, with_camera = True)
        return prediction

    def _remove_od(self, orig_im_file, prediction):
        # Remove FPs due to OD
        assert(path.exists(orig_im_file)), "Original image does not exist"
        assert (self._image.size > 0), "No image has been analyzed"

        # detect the OD and rescale 
        od = DetectOD(orig_im_file, self._masks_dir)
        ctr = od.locate_disk()

        # flood-fill to mask off the disk. The mask needs to be 2 pixels
        # larger than the image
        h, w = self._mask.shape[0] + 2, self._mask.shape[1] + 2
        mask = cv2.bitwise_not(cv2.resize(self._mask, (w, h)))

        # connectivit is 8 neighbors, fill mask with 255
        flags = 8 | ( 255 << 8 ) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(prediction, mask, ctr, 0, 0, 0, flags)

        self.display_current(prediction)

        return prediction

    def refine_prediction(self, orig_im_file, prediction):
        '''
        Removes false-positives
        orig_im_file - the original imag
        '''
        prediction = self._remove_od(orig_im_file, prediction)
        return prediction