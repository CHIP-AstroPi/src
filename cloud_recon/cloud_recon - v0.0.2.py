#from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import scipy.misc


def cut_image(img):
        w,h,_ = img.shape
        P = [int(w/2),int(h/2)]
        perc_h = 65
        perc_w = 65
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 

        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]

def cloud_recon(img):
    img = cv.imread(img)    #read and cut img
    
    img = cut_image(img)

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)    #convert rgb
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)    #convert hsv

    s = hsv[:, :, 1]
    ret, thresh = cv.threshold(s, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)  #setup threshold 

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]  #setup countours
    cv.drawContours(img, contours, -1, (0,255,0), 2) #draw countours on img

    x, y, w, h = cv.boundingRect(contours[0])
    thresh[y:y+h, x:x+w] = 255 - thresh[y:y+h, x:x+w]
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]

    for c in contours:  #fill countours inner border
        if cv.contourArea(c) > 4:  
            cv.drawContours(img, [c], -1, (255, 0, 0), thickness=cv.FILLED)

    
    #total pixel calc
    total_pixel = img.shape[0] * img.shape[1]
    print(f"total pixel: {total_pixel}")

    #inner border pixel fill
    sought = [255,0,0]
    red = np.count_nonzero(np.all(img==sought,axis=2))
    print(f"red pixel: {red}")

    #perc of red pixel in img
    perc = (red*100)/total_pixel
    perc = "%.3f" % perc
    print(f"perc: {perc}%")


if __name__ == "__main__":
    for k in range (1,28):
        cloud_recon(f"./dataset_image/train{k}.jpg")
        print("\n")
