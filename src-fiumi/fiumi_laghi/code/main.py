import cv2 as cv
import numpy as np
import math

#colori

GREEN = (0,255,0)


#inzio funzioni

def fractal_dimension(img,contours):    
    """Comput fractal dimension of coastlines

    Given an image of a coastline and a list of contours,
    this function comput the fractal value of the coastline
    using a boxcounting algorithm.

    """
    
    w,h = img.shape[:2] #get width and height from the image
    img = np.zeros((w,h,1), np.uint8)   #create a black background image with same dimensions

    #draw contours on the image
    cv.drawContours(img, contours, -1, (255, 255, 255), 1)

    while w % 5 > 0:    #adjust dimensions so it get perfect 5x5 pixel squares
        w = w-1
    while h % 5 > 0:
        h = h-1

    higher_scale = 0
    for r in range(int(h/5)):
        for c in range(int(w/5)):
            if cv.countNonZero(img[r*5:r*5+5,c*5:c*5+5]) > 0:
                higher_scale = higher_scale + 1

    #INCREASE RESOLUTION OF THE GRID
    while w % 2 > 0:     #adjust dimensions so it get perfect 2x2 pixel squares
        w = w-1
    while h % 2 > 0:
        h = h-1

    lower_scale = 0

    for r in range(int(h/2)):
        for c in range(int(w/2)):
            if cv.countNonZero(img[r*2:r*2+2,c*2:c*2+2]) > 0:
                lower_scale = lower_scale + 1

    return math.log(lower_scale/higher_scale,10) / math.log(2,10)   #fractal dimension 
                


def cut_image(img,top = 65,left = 65):
        perc_h = top
        perc_w = left
        w,h = img.shape[:2]
        P = [int(w/2),int(h/2)]
        
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 

        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]



def adaptive_threshold(Ray_img, Color_img):
    th = cv.adaptiveThreshold(Ray_img,100,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,901, -5)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    contours = [c for c in contours if cv.arcLength(c,False) >400 ]
    lowerbound = 0
    upperbound = 1

    w,h = Color_img.shape[:2]
    fractal_img = np.zeros((w, h, 1), dtype = "uint8")
    for cont in contours:
        p = cv.arcLength(cont, True)
        a = cv.contourArea(cont)
        d = abs(p/a - 1) if a > 0 else None
        if d and lowerbound <= d <= upperbound:
            cv.drawContours(Color_img, [cont],-1, GREEN, 1)

    print(fractal_dimension(fractal_img,contours))
            
    return Color_img
    
def main():
    
    for i in range(1,12):
        img = cv.imread(f"C:/Users/lucab/Desktop/repo/astro_pi/src/src-fiumi/fiumi_laghi/code/dataset_image/Image{i}.jpg")
        Cut_Img_Color = cut_image(img,55,55)

        gray_img = cv.cvtColor(Cut_Img_Color, cv.COLOR_BGR2GRAY)

        cv.imwrite(f"C:/Users/lucab/Desktop/repo/astro_pi/src/src-fiumi/fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", adaptive_threshold(gray_img,Cut_Img_Color))

if __name__ == "__main__":
    main()
