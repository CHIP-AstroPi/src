import cv2 as cv
import numpy as np
import math

#colori

GREEN = (0,255,0)


#inzio funzioni

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
            cv.drawContours(fractal_img,[cont],-1, (255,255,255), 1)

    fractal_dimension(fractal_img)
            
    return Color_img
    
def main():
    
    for i in range(1,12):
        img = cv.imread(f"C:/Users/lucab/Desktop/astropi/src/src-fiumi/fiumi_laghi/code/dataset_image/Image{i}.jpg")
        Cut_Img_Color = cut_image(img,55,55)

        gray_img = cv.cvtColor(Cut_Img_Color, cv.COLOR_BGR2GRAY)

        cv.imwrite(f"C:/Users/lucab/Desktop/astropi/src/src-fiumi/fiumi_laghi/code/data_elab/elab{i}_adaptive.jpg", adaptive_threshold(gray_img,Cut_Img_Color))

if __name__ == "__main__":
    main()
