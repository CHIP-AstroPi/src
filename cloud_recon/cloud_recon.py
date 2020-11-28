import sys,cv2
import numpy as np

img = cv2.imread('/home/rdfilippo/Desktop/Scuola/AstroPi/src/cloud_recon/images/train14.jpg', 1)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

white = np.array([255,255,255])
lowerBound = np.array([30,30,30])

mask = cv2.inRange(hsv, lowerBound, white)

res = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("/home/rdfilippo/Desktop/Scuola/AstroPi/src/cloud_recon/images/train14-1.jpg", res)

cv2.imshow("mywindow", res)

cv2.waitKey(0)