import cv2 as cv
import numpy as np

#colori

GREEN = (0,255,0)


#inzio funzioni

def cut_image(img,top = 65,left = 65):
        perc_h = top
        perc_w = left
        w,h,_ = img.shape
        P = [int(w/2),int(h/2)]
        
        padding_top = int((P[1] * perc_h) / 100)
        padding_side = int((P[0] * perc_w) / 100 ) 

        return img[P[0] - padding_side:P[0]+ padding_side,P[1]- padding_top:P[1]+padding_top]


def whitePrecentage(path):
        img = cv.imread(path)

        height, width = img.shape[:2]

        return ((np.count_nonzero(img)/3) * 100)/ (height*width)

def color_detection(img):   #MUST BE HSV IMAGE
    """
    lower = np.array([90,100,20])
    upper = np.array([125,255,255])"""

    threshold_type = 1
    threashold_value = 134
    max_binary_value = 255
    _, mask = cv.threshold(img, threashold_value, max_binary_value, threshold_type)

    return mask

def adaptive_threshold(Ray_img, Color_img):
    th = cv.adaptiveThreshold(Ray_img,100,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,901, -5)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    contours = [c for c in contours if cv.arcLength(c,False) >400 ]
    lowerbound = 0
    upperbound = 1

    for cont in contours:
        p = cv.arcLength(cont, True)
        a = cv.contourArea(cont)
        d = abs(p/a - 1) if a > 0 else None
        if d and lowerbound <= d <= upperbound:
            cv.drawContours(Color_img, [cont],-1, GREEN, 1)
            
    return Color_img, 
"""    cv.drawContours(img, contours, -1, (0, 255, 0), 3 ) 
        opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
    dilation = cv.dilate(opening,kernel,iterations=1)
    cv.fillPoly(dilation, pts =contours, color=(255,255,255)) """
    
#fine funzioni
    


def main():
    
    for i in range(1,12):
        img = cv.imread(f"src-fiumi/fiumi_laghi/code/dataset_image/Image{i}.jpg")
        Cut_Img_Color = cut_image(img,55,55)

        
        gray_img = cv.cvtColor(Cut_Img_Color, cv.COLOR_BGR2GRAY)
        #hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        #rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        #OLD METHOD
        
        #cv.imwrite(f"src-fiumi/fiumi_laghi/code/data_elab/elab{i}.jpg", color_detection(gray_img))

        #ADAPTIVE THRESHOLD

        cv.imwrite(f"src-fiumi/fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", adaptive_threshold(gray_img,Cut_Img_Color))

        #cv.imwrite(f"fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", hsv_img)





if __name__ == "__main__":
    main()
