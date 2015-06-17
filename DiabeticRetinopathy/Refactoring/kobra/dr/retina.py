import numpy as np
import cv2
from kobra.imaging import show_images

def display_contours(image, contours, color = (255, 0, 0), thickness = -1, title = None):
    imShow = image.copy()
    for i in range(0, len(contours)):
        cv2.drawContours(imShow, contours, i, color, thickness)
    show_images([imShow], scale=0.7, titles=title)


def createMask((rows, cols), hull):
    # black image
    mask = np.zeros((rows, cols), dtype=np.uint8)
    # blit our contours onto it in white color
    cv2.drawContours(mask, [hull], 0, 255, -1)

    return mask

def find_eye(image, thresh = 4):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge finder
    edges = np.array([])
    edges = cv2.Canny(gray, thresh, thresh * 3, edges)

    # Find contours
    # second output is hierarchy - we are not interested in it.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Now let's get only what we need out of it
    hull_contours = cv2.convexHull(np.vstack(np.array(contours)))
    hull = np.vstack(hull_contours)
    
    mask = createMask(image.shape[0:2], hull)
    
    # returning the hull to illustrate a few issues below
    return mask, hull
