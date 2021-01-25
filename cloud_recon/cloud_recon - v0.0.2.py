#from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import scipy.misc


def cut_image(img):
        w,h = img.shape[:2]
        P = [int(w/2),int(h/2)]
        perc_h = 65
        perc_w = 65
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 

        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]

def fractal_dimension(img):     #img must be a black gb image with contours drawn on it in any colors except black
    w,h = img.shape[:2]
    print(f"h:{h}, w:{w}")
    
    with open("fractal_value.txt","a") as f:
        while w % 5 > 0:
            w = w-1
        while h % 5 > 0:
            h = h-1
        print(f"h:{h}, w:{w}")
        rows, cols = int(h/5), int(w/5)



        start_box_counting = 0
        for r in range(rows):
            for c in range(cols):
                if cv.countNonZero(img[r*5:r*5+5,c*5:c*5+5]) > 0:
                    start_box_counting = start_box_counting + 1

        while w % 2 > 0:
            w = w-1
        while h % 2 > 0:
            h = h-1
        print(f"h:{h}, w:{w}")
        rows, cols = int(h/2), int(w/2)
        end_box_counting = 0
        for r in range(rows):
            for c in range(cols):
                if cv.countNonZero(img[r*2:r*2+2,c*2:c*2+2]) > 0:
                    end_box_counting = end_box_counting + 1

        print(f"log({end_box_counting}/{start_box_counting},10)/log({math.log(2,10)})\n")

        fractal_dim = math.log(end_box_counting/start_box_counting,10) / math.log(2,10)

        f.write(f"log({end_box_counting/start_box_counting,10})/log(2) = {fractal_dim}\n")
        
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

    w,h = img.shape[:2]
    fractal_img = np.zeros((w,h,1), np.uint8)
    for c in contours:  #fill countours inner border
        if cv.contourArea(c) > 4:  
            cv.drawContours(img, [c], -1, (255, 0, 0), thickness=cv.FILLED)
            cv.drawContours(img, [c], -1, (255, 255, 255), 1)
    
    cv.imshow("fractal contour",fractal_img)
    cv.waitKey()

    
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
