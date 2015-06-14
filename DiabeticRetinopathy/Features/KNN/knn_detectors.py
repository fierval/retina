import numpy as np
from regions_detect_knn import KNeighborsRegions, Labels
from kobra import enum

class DrusenDetect(KNeighborsRegions):
    def __init__(self, root, annotations, masks_dir, orig_path, n_neighbors = 3):

        KNeighborsRegions.__init__(self, root, annotations, masks_dir, orig_path, n_neighbors)
        self._labels =  [Labels.Drusen, Labels.Drusen, Labels.Background, Labels.Background, 
                    Labels.CameraHue, Labels.CameraHue, 
                    Labels.Outside, Labels.Outside, Labels.OD, Labels.OD]

    def display_current(self, prediction, with_camera = False):
        self.display_artifact(prediction, Labels.Drusen, (0, 255, 0), "Drusen")
        self.display_artifact(prediction, Labels.Background, (0, 0, 255), "Haemorages")

class HaemDetect(KNeighborsRegions):
    def __init__(self, root, annotations, masks_dir, orig_path, n_neighbors = 3):

        KNeighborsRegions.__init__(self, root, annotations, masks_dir, orig_path, n_neighbors)
        self._labels =  [Labels.Blood, Labels.Blood, Labels.Blood, 
                        Labels.Bright, Labels.Bright, Labels.Bright, 
                        Labels.Dark, Labels.Dark, Labels.Dark]

    def display_current(self, prediction, with_camera = False):
        self.display_artifact(prediction, Labels.Blood, (0, 255, 0), "Blood")

