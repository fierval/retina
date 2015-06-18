import numpy as np
from regions_detect_knn import KNeighborsRegions, Labels
import mahotas as mh
from kobra.imaging import max_labelled_region
import cv2

root = '/kaggle/retina/train/labelled'
annotations = 'annotations.txt'
masks_dir = '/kaggle/retina/train/masks'
im_file = '4/16_left.jpeg'
orig_path = '/kaggle/retina/train/sample'

class DarkBrightDetector(KNeighborsRegions):
    '''
    Detect bright/dark spots.
    Drusen/Exudates (bright) and haemorages/aneurisms (dark) detected
    '''
    def __init__(self, root, im_file, annotations, masks_dir, n_neighbors = 3):

        KNeighborsRegions.__init__(self, root, im_file, annotations, masks_dir, n_neighbors)
        self._prediction = np.array([])
        # Labels match the structure of the annotations file:
        '''
        Averages of the pixels values of all images:
        Annotations contain:
        position: meaning 
        0 - 1: drusen/exudates 
        2 - 3: texture 
        4 - 5: Camera effects
        6 - 7: Outside/black
        8 - 9: Optic Disc
        '''

        self._labels =  [Labels.Drusen, Labels.Drusen, Labels.Background, Labels.Background, 
                    Labels.CameraHue, Labels.CameraHue, 
                    Labels.Outside, Labels.Outside, Labels.OD, Labels.OD]

    def display_current(self, prediction, with_camera = False):
        self.display_artifact(prediction, Labels.Drusen, (0, 255, 0), "Drusen")
        self.display_artifact(prediction, Labels.Outside, (0, 0, 255), "Haemorages")

    def _get_predicted_region(self, label):
        region = self._prediction.copy()
        region [region != label] = 0
        return region

    def find_bright_regions(self, thresh = 0.005, abs_area = 50):
        '''
        Finds suspicious bright regions, refines the prediction by:
        - Removing areas of large size: >= image * thresh
        - and small size: <= abs_area
        '''
        assert( 0 < thresh < 1), "Threshold is in (0, 1)"
        if self._prediction.size == 0:
            self._prediction = self.analyze_image()

        # cutoff area
        area = cv2.countNonZero(self._mask) * thresh

        # get bright regions and combine them
        ods = self._get_predicted_region(Labels.OD)
        drusen = self._get_predicted_region(Labels.Drusen)
        combined = ods + drusen
              
        # relabel the combined regions & calculate large removable regions
        # this will remove OD and related effects
        labelled, n = mh.label(combined, Bc = np.ones((9, 9)))
        sizes = mh.labeled.labeled_size(labelled)[1:]
        to_remove = np.argwhere(sizes >= area) + 1        

        #compute small regions to be removed
        to_remove_small = np.argwhere(sizes <= abs_area) + 1
        to_remove = np.r_[to_remove, to_remove_small]

        # remove large areas from prediction matrix
        for i in to_remove:
            self._prediction [labelled == i] = Labels.Masked

        # re-label the rest of the OD artifacts with Drusen
        #self._prediction[self._prediction == Labels.OD] = Labels.Drusen
        self.display_current(self._prediction)

    def get_bright_regions(self):
        '''
        Mask off regions immediately adjecent to the OD
        '''

        # Find the largest "OD" region

        # Initial labelling
        if self._prediction.size == 0:
            self._prediction = self.analyze_image()

        ods = self._prediction.copy()

        # find maximum size "OD" region
        # this should be the OD itself
        ods[ods != Labels.OD] = 0
        ods, _ = mh.label(ods, Bc = np.ones((9, 9)))

        # label of the region of maximum size
        # 0-th region is always the background, so we leave it out
        od_region = max_labelled_region(ods)

        # mask out everything except the OD
        ods [ ods != od_region] = 0

        # combine OD with all bright regions
        bright = self._prediction.copy()
        bright [ bright != Labels.Drusen] = 0
        bright = bright + ods

        # maximum connected region has to be eliminated
        od_bright, _ = mh.label(bright, Bc = np.ones((9, 9)))
        od_bright_max = max_labelled_region(od_bright)
        od_bright [od_bright != od_bright_max] = 0

        # in our mask remove the region corresponding to
        # OD + connected artifacts
        self._mask [od_bright != 0] = 0
        self._prediction [od_bright != 0] = Labels.Masked

        self.display_current(self._prediction)