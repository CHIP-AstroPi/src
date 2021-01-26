import cv2 as cv
import numpy as np
from random import randint


GREEN = (0,255,0)

def cut_image(img,top = 65,left = 65):
        perc_h = top
        perc_w = left
        w,h,_ = img.shape
        P = [int(w/2),int(h/2)]
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 
        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]


def whitePrecentage(img):
        height, width = img.shape[:]
        return ((np.sum(img == 255)) * 100)/ (height*width)

def Is_insland_ghost(img,cont):
    
    c = cv.boundingRect(cont)
    x,y,w,h = c
    n = img[y:y+h, x:x+w]
    n = cv.cvtColor(n,cv.COLOR_BGR2GRAY)
    _, n=  cv.threshold(n, 134,255,1)

    if whitePrecentage(n) < 80.0:
            
        return True
    else:
        return False

def adaptive_threshold(Ray_img, Color_img):
    th = cv.adaptiveThreshold(Ray_img,100,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,1101, -6)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    contours = [c for c in contours if cv.arcLength(c,False) >500 ]
    lowerbound = 0
    upperbound = 1
    
    for i,cont in enumerate(contours):

        p = cv.arcLength(cont, True)
        a = cv.contourArea(cont)
        d = abs(p/a - 1) if a > 0 else None
        if d and lowerbound <= d <= upperbound  :

            cv.drawContours(Color_img, [cont],-1, (randint(10,255),randint(100,255),randint(100,200)), 2)
            if Is_insland_ghost(Color_img,cont):
                contours.pop(i)
        
        
    #per jom: countours returna in un array tutti i contorni di un immagine. per adesso elimina solo dal disegno e non dall'array

    return Color_img
    
def main():
    
    for i in range(1,37):
        img = cv.imread(f"C:/Users/fabri/Desktop/scuola/project/src/src-fiumi/fiumi_laghi/code/dataset_image/Image{i}.jpg")
        Cut_Img_Color = cut_image(img,55,55)
        gray_img = cv.cvtColor(Cut_Img_Color, cv.COLOR_BGR2GRAY)
        cv.imwrite(f"C:/Users/fabri/Desktop/scuola/project/src/src-fiumi/fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", adaptive_threshold(gray_img,Cut_Img_Color))
                     

if __name__ == "__main__":
    main()
