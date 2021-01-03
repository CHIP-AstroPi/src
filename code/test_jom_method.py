import cv2 as cv
import numpy as np
from pathlib import Path, PurePath
from config import PATH_IMAGE, PATH_ELAB


max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'
dst = None
Image_name = "image2.jpg"

cartella_Image: Path = Path(PATH_IMAGE)
cartella_elab : Path = Path(PATH_ELAB)

file_Image : PurePath= cartella_Image / Image_name

Img = cv.imread(str(file_Image))

def Threshold_Demo():
    pass

src_gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
cv.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold_Demo)

    
while True:
        threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
        threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
        _, dst = cv.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        cv.imshow(window_name, dst)

        key = cv.waitKey(1)
        if key == 27:
            break

cv.imwrite(f"{str(PATH_ELAB)}elab_jom_{Image_name}",dst)
cv.destroyAllWindows()