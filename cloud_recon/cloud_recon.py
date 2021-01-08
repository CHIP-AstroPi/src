import cv2 as cv
import numpy as np

img = cv.imread('./images/train10.jpg')
#img = img[:, 350:1500]

plt.imshow(imRGB)
plt.title('original')
plt.show()

#img = cv.GaussianBlur(img, (5,5), 0)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2HSV))
plt.title('image')
plt.show()

white = np.array([255,255,255])
lowerBound = np.array([30,10,30])

mask = cv.inRange(hsv, lowerBound, white)

res = cv.bitwise_and(img, img, mask=mask)

plt.imshow(cv.bitwise_and(img, img, mask=mask))
plt.title('elaborated')
plt.show()

#count of total pixel
total = res.shape[0] * res.shape[1]
print(f"total pixel: {total}")

#count of black pixel
sought = [0,0,0]
black  = np.count_nonzero(np.all(res==sought,axis=2))
print(f"black pixel: {black}")

#perc of black pixels
perc = (black*100)/total
perc = "%.3f" % perc
print(f"perc: {perc}%")