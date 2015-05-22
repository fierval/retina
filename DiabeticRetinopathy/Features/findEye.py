import numpy as np
import cv2
from os import path
import os
import matplotlib.pyplot as plt

root_path = "/kaggle/retina"

# train/test directories
train_path = path.join(root_path, 'train')
test_path = path.join(root_path, 'test')
raw_train = path.join(train_path, 'raw')
raw_test = path.join(test_path, 'raw')
sample_train = path.join(train_path, 'sample')

def nothing(*arg):
    pass

def createMask((rows, cols), hull):
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.drawContours(mask, [hull], 0, 255, -1)
    return mask

def loadMask(file_path, size):
    return np.fromfile(file_path).reshape(size, size) 

thresh = 4
size = 256

processed_path = path.join(train_path, "dbg")
sample_path = sample_train
file_name = "360_right.jpeg"

#sample_image_path = path.join(processed_path, "320_right.jpeg")
#sample_image_path = path.join(processed_path, "11182_right.jpeg")
#sample_image_path = path.join(processed_path, "21118_left.jpeg")
#sample_image_path = path.join(processed_path, "32253_right.jpeg")
#sample_image_path = path.join(processed_path, "9340_left.jpeg")
#sample_image_path = path.join(processed_path, "360_right.jpeg")
#sample_image_path = path.join(processed_path, "2047_right.jpeg") # threshold 73
#sample_image_path = path.join(processed_path, "2351_right.jpeg") # threshold 60
#sample_image_path = path.join(processed_path, "240_left.jpeg")
#sample_image_path = path.join(processed_path, "16_right.jpeg") 
sample_image_path = path.join(sample_path, file_name)  #threshold 22
proc_image_path = path.join(processed_path, file_name)

#sample_image_path = path.join(processed_path, "5784_right.jpeg") 

srcImage = cv2.resize(cv2.imread(sample_image_path), (size, size))
srcGrey = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
srcGrey = cv2.resize(cv2.GaussianBlur(srcGrey, (7, 7), 30), (size, size))

destImg = cv2.imread(proc_image_path)

cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", "Source", thresh, thresh * 6, nothing)

cv2.namedWindow("Proc", cv2.WINDOW_NORMAL)

while True:
    src = np.copy(srcImage)
    srcG = np.copy(srcGrey)
    dest = np.copy(destImg)

    thresh = cv2.getTrackbarPos("Threshold", "Source")

    edges = np.array([])
    edges = cv2.Canny(srcG, thresh, thresh * 3, edges, 3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = cv2.approxPolyDP(contours[0], 5, True)
    for cnt in contours[1:]:
        polys = np.vstack((polys, cv2.approxPolyDP(cnt, 5, True)))

    hull_contours = cv2.convexHull(polys)
    hull = np.vstack(hull_contours)
    hull_area = cv2.contourArea(hull)

    for i in range(0, len(contours)):
        cv2.drawContours(src, contours, i, (255, 0, 0), 1)

    cv2.drawContours(src, [hull], 0, (0, 0, 255), 1)
    mask = createMask(src.shape[0:2], hull)
    cv2.namedWindow("Mask")
    cv2.imshow("Mask", mask)

    cv2.drawContours(dest, [hull], 0, (0, 0, 255), 2)
    cv2.imshow("Proc", dest)
    cv2.imshow("Source", src)
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()

