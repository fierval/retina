# image histogram remapping

import cv2
from os import path
import os
import color_transfer as ct

im_path = "/Users/Boris/Dropbox/Kaggle/retina/train/sample"
source_image_path = path.join(im_path, "2551_right.jpeg")
target_image_path = path.join(im_path, "16_left.jpeg")

source_image_orig = cv2.imread(source_image_path)
target_image_orig = cv2.imread(target_image_path)

srcImg = cv2.resize(cv2.pyrDown(cv2.pyrDown(source_image_orig)), (500, 500))
tgtImg = cv2.resize(cv2.pyrDown(cv2.pyrDown(target_image_orig)), (500, 500))

trnsfImg = ct.color_transfer(srcImg, tgtImg)

cv2.namedWindow("Source")
cv2.namedWindow("Target")
cv2.namedWindow("Transfered")

cv2.imshow("Source", srcImg)
cv2.imshow("Target", tgtImg)
cv2.imshow("Transfered", trnsfImg)

cv2.waitKey(0)


cv2.destroyAllWindows()
