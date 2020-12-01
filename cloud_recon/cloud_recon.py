import cv2
import numpy as np


def recon(image):
        img = cv2.imread(image, 1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    white = np.array([255,255,255])
    lowerBound = np.array([30,30,30])

    mask = cv2.inRange(hsv, lowerBound, white)

    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(image, res)

    cv2.waitKey(0)