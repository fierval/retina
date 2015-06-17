import numpy as np
from blood_detection import KNeighborsRegions, Labels
from kobra import enum

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
