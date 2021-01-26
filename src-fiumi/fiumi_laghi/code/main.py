import cv2 as cv
import numpy as np
from random import randint
#colori

GREEN = (0,255,0)
RED = (0,0,255)

#inzio funzioni


def cut_image(img,top = 65,left = 65):
        perc_h = top
        perc_w = left
        w,h,_ = img.shape
        P = [int(w/2),int(h/2)]
        
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 

        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]


def whitePrecentage(img):
        #img = cv.imread(path)
        height, width = img.shape[:]
        
        return ((np.sum(img == 255)) * 100)/ (height*width)

def color_detection(img):   #MUST BE HSV IMAGE
    """
    lower = np.array([90,100,20])
    upper = np.array([125,255,255])"""

    threshold_type = 1
    threashold_value = 134
    max_binary_value = 255
    _, mask = cv.threshold(img, threashold_value, max_binary_value, threshold_type)

    return mask

def Is_insland_ghost(img,cont,i):
    
    c = cv.boundingRect(cont)
    x,y,w,h = c
    n = img[y:y+h, x:x+w]
    
    n = cv.cvtColor(n,cv.COLOR_BGR2GRAY)
    _, n=  cv.threshold(n, 134,255,1)

    

    cv.imwrite(f"C:/Users/fabri/Desktop/scuola/project/src/src-fiumi/fiumi_laghi/code/data_elab/cut/image_{i}_{w}_cut.jpg",n)


    print("image"+str(i)+": "+str(whitePrecentage(n)))
    if whitePrecentage(n) < 80.0:
        cv.drawContours(cv.cvtColor(n,cv.COLOR_GRAY2BGR), [cont],-1, GREEN, 1)     
        return True
    else:
        return False

def adaptive_threshold(Ray_img, Color_img,i):
    
    th = cv.adaptiveThreshold(Ray_img,100,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,1101, -6)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    contours = [c for c in contours if cv.arcLength(c,False) >500 ]
    lowerbound = 0
    upperbound = 1
    
    for cont in contours:
        p = cv.arcLength(cont, True)
        a = cv.contourArea(cont)
        d = abs(p/a - 1) if a > 0 else None

        if d and lowerbound <= d <= upperbound and Is_insland_ghost(Color_img,cont,i):
            cv.drawContours(Color_img, [cont],-1, (randint(10,255),randint(100,255),randint(100,200)), 2)

    
    



    return Color_img
"""    cv.drawContours(img, contours, -1, (0, 255, 0), 3 ) 
        opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
    dilation = cv.dilate(opening,kernel,iterations=1)
    cv.fillPoly(dilation, pts =contours, color=(255,255,255)) """
    
#fine funzioni
    


def main():
    
    for i in range(1,37):
        img = cv.imread(f"C:/Users/fabri/Desktop/scuola/project/src/src-fiumi/fiumi_laghi/code/dataset_image/Image{i}.jpg")
        Cut_Img_Color = cut_image(img,55,55)

        gray_img = cv.cvtColor(Cut_Img_Color, cv.COLOR_BGR2GRAY)

        cv.imwrite(f"C:/Users/fabri/Desktop/scuola/project/src/src-fiumi/fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", adaptive_threshold(gray_img,Cut_Img_Color,i))
                     



if __name__ == "__main__":
    main()
