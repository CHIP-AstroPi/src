""" 
autori : smilefabri, scylla 
versione : 1.0.1

"""



import cv2 as cv
import numpy as np
from pathlib import Path, PurePath
from config import PATH_IMAGE, PATH_ELAB
import time

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'
dst = None
Image_name = "image3.jpg"

cartella_Image: Path = Path(PATH_IMAGE)
cartella_elab : Path = Path(PATH_ELAB)

file_Image : str= str(cartella_Image / Image_name)

Img = cv.imread(str(file_Image))

def Threshold_Demo():
    pass


def whitePrecentage(path):
    img = cv.imread(path)

    height, width = img.shape[:2]
    white = 0
    for y in range(height):
        for x in range(width):
            if img[y][x][0] != 0 and img[y][x][1] != 0 and img[y][x][2] != 0 :
                white += 1
    
    return((white*100)/(width*height))

    
src_gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
#cv.namedWindow(window_name)
#cv.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
#cv.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold_Demo)

start_time = time.time()

while True:
    threshold_type = 1# cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = 134# cv.getTrackbarPos(trackbar_value, window_name)
    _, dst=  cv.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    cv.imshow(window_name, dst)
        



    key = cv.waitKey(1)
    if key == 27:
        cv.imwrite(f"{str(PATH_ELAB)}elab_jom_{Image_name}",dst)
        print("percentuale di mare: "+ str(whitePrecentage(f"{str(PATH_ELAB)}elab_jom_{Image_name}")))
        break



cv.destroyAllWindows()
print(f"---{time.time() - start_time} secondi---")