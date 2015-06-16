from train_files import TrainFiles
from SupervisedLearning import SKSupervisedLearning
from od_detection import DetectOD
from image_reader import ImageReader
import cv2

def enum(**enums):
    return type('Enum', (), enums)
