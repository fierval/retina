import numpy as np
import pandas as pd
import cv2
import os
from os import path

cold_path = "/users/boris/Dropbox/Shared/Retina/coldMap_gaussBlur"
warm_path = "/users/boris/Dropbox/Shared/Retina/warmMap_gaussBlur"

fileName_4 = "4/16_left"
fileName_0 = "0/10_left"
ext = ".jpeg"

def make_file_name(in_path, fileName):
    return path.join(in_path, fileName + ext)

img_cold_4 = cv2.imread(make_file_name(cold_path, fileName_4), cv2.IMREAD_COLOR)
img_cold_0 = cv2.imread(make_file_name(cold_path, fileName_0), cv2.IMREAD_COLOR)

img_warm_0 = cv2.imread(make_file_name(warm_path, fileName_0), cv2.IMREAD_COLOR)
img_warm_4 = cv2.imread(make_file_name(warm_path, fileName_4), cv2.IMREAD_COLOR)

imgs = [img_cold_0, img_cold_4, img_warm_0, img_warm_4]

imgs_gray = map(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), imgs)

detector = cv2.SimpleBlobDetector()

keyp = map(lambda im: detector.detect(im), imgs_gray)

im_with_keypoints = map(lambda (im, k): cv2.drawKeypoints(im, k, np.array([]), (0, 0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), zip(imgs, keyp))

cv2.namedWindow("keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("keypoints",  im_with_keypoints[0])
cv2.waitKey(0)