# image histogram remapping

import cv2
from os import path
import os
import color_transfer as ct
from skimage import util
from skimage.draw import circle

def circularcrop(img, border, threshold, threshold1):
    s = np.sum(img, axis=2)
    cols = np.sum(s, axis=0) > threshold  
    rows = np.sum(s, axis=1) > threshold
    
    height = rows.shape[0]
    width = cols.shape[0]

    x_min = np.argmax(cols[0:width])
    x_max = width/2 + np.argmin(cols[width/2:width-1])
    y_min = np.argmax(rows[0:height/2])
    y_max = np.argmin(cols[height/2:height-1])
    y_max = height/2 + y_max if y_max > 0 else height

    radius = (x_max - x_min)/2
    center_x = x_min + radius
    center_y = y_min + radius # the default case (if y_min != 0)
    if y_min == 0: # the upper side is cropped
        if height - y_max > 0: # lower border is not 0
            center_y = y_max - radius
        else:
            upper_line_width = np.sum(s[0,:] > threshold1) # threshold for single line
            center_y = math.sqrt( radius**2 - (upper_line_width/2)**2)
    radius1 = radius - border    
    
    mask = np.zeros(img.shape[0:2])
    rr, cc = circle(center_y, center_x, radius1, img.shape)
    mask[rr, cc] = 1
    img[:,:,0] *= mask
    img[:,:,1] *= mask
    img[:,:,2] *= mask 
    
    x_borders = (center_x - radius1, img.shape[1] - center_x - radius1)
    y_borders = (max(center_y - radius1,0), max(img.shape[0] - center_y - radius1, 0))

    imgres = util.crop(img, (y_borders, x_borders,  (0,0)))
    maskT = util.crop(mask, (y_borders, x_borders))

    border_pixels = np.sum(1 - maskT)
    
    return imgres, maskT, center_x, center_y, radius

im_path = "/Kaggle/retina/train/dbg"
source_image_path = path.join(im_path, "360_right.jpeg")
target_image_path = path.join(im_path, "6550_left.jpeg")

srcImg = cv2.imread(source_image_path)

srcImg = circularcrop(srcImg, 200, 2000, 100)
cv2.namedWindow("Source")
cv2.imshow("Source", srcImg)

tgtImg = cv2.imread(target_image_path)

srcImg, cv2.pyrDown(cv2.pyrDown(srcImg))
tgtImg = cv2.pyrDown(cv2.pyrDown(tgtImg))

trnsfImg = ct.color_transfer(srcImg, tgtImg)

cv2.namedWindow("Target")
cv2.namedWindow("Transfered")

cv2.imshow("Target", tgtImg)
cv2.imshow("Transfered", trnsfImg)

cv2.waitKey(0)


cv2.destroyAllWindows()
