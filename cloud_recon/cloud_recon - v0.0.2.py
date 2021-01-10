from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import scipy.misc

def cloud_recon(img):
    img = cv.imread(img)
    img = img[:, 350:1500]

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

    s = hsv[:, :, 1]
    ret, thresh = cv.threshold(s, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2] 
    cv.drawContours(img, contours, -1, (0,255,0), 2)

    x, y, w, h = cv.boundingRect(contours[0])
    thresh[y:y+h, x:x+w] = 255 - thresh[y:y+h, x:x+w]
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]

    for c in contours:
        if cv.contourArea(c) > 4:  
            cv.drawContours(img, [c], -1, (255, 0, 0), thickness=cv.FILLED)

    total_pixel = img.shape[0] * img.shape[1]
    print(f"total pixel: {total_pixel}")

    sought = [255,0,0]
    red = np.count_nonzero(np.all(img==sought,axis=2))
    print(f"red pixel: {red}")

    perc = (red*100)/total_pixel
    perc = "%.3f" % perc
    print(f"perc: {perc}%")


if __name__ == "__main__":
    cloud_recon('./images/train9.jpg')