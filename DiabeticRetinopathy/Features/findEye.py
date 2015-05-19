import numpy as np
import cv2
from os import path
import os
import matplotlib.pyplot as plt

import environment as evt
def nothing(*arg):
    pass

thresh = 70

processed_path = path.join(evt.train_path, "dbg")
sample_image_path = path.join(processed_path, "360_right.jpeg")

srcImage = cv2.imread(sample_image_path)
srcGrey = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
srcGrey = cv2.GaussianBlur(srcGrey, (7, 7), 20)

cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", "Source", thresh / 2, thresh * 4, nothing)

while True:
    src = np.copy(srcImage)
    srcG = np.copy(srcGrey)

    ## Try Hough circles
    #circles = cv2.HoughCircles(srcG, 3, 1, 10)
    #for (x, y, r) in circles:
    #    cv2.circle(src, (x, y), r, (0, 255, 0), 2)
    #    cv2.rectangle(src, (x-3, y-3), (x+3, y+3), (0, 255, 0))

    thresh = cv2.getTrackbarPos("Threshold", "Source")

    edges = np.array([])
    edges = cv2.Canny(srcG, thresh, thresh * 3, edges, 3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = cv2.approxPolyDP(contours[0], 5, True)
    for cnt in contours[1:]:
        polys = np.vstack((polys, cv2.approxPolyDP(cnt, 5, True)))

        #x, y, r = int(x), int(y), int(r)
        #cv2.circle(src, (x, y), r, (0, 255, 0), 2)
    hull = np.vstack(cv2.convexHull(polys))
    rotatedRect = cv2.minAreaRect(hull)

    ext_color = (255, 255, 255)
    for i in range(0, len(contours)):
        cv2.drawContours(src, contours, i, ext_color, 1)

    cv2.imshow("Source", src)
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()

