import cv2 as cv
import numpy as np
from pathlib import Path, PurePath

from numpy.core.defchararray import upper
from config import PATH_IMAGE, PATH_ELAB
import time

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'
dst = None
Image_name = "image16.jpg"

cartella_Image: Path = Path(PATH_IMAGE)
cartella_elab : Path = Path(PATH_ELAB)

file_Image : str= str(cartella_Image / Image_name)

Img = cv.imread(str(file_Image))

#inizio funzioni
def getMouseCoords(event, x, y, flags, param):
    if  event == cv.EVENT_LBUTTONDBLCLK:
        print(f"X: {x}     |     Y:{y}")


def whitePrecentage(path):
        img = cv.imread(path)

        height, width = img.shape[:2]

        return ((np.count_nonzero(img)/3) * 100)/ (height*width)

#fine funzioni


while True:        
        src_gray = cv.cvtColor(Img, cv.COLOR_BGR2HSV)

        start_time = time.time()

        threshold_type = 1
        threashold_value = 184
        _, mask = cv.threshold()

        cv.imshow("maschera",mask)

        key = cv.waitKey(1)
        if key == 27:
                cv.imwrite(f"{str(PATH_ELAB)}elab_jom_{Image_name}",mask)
                #print("percentuale di mare: "+ str(whitePrecentage(f"{str(PATH_ELAB)}elab_jom_{Image_name}")))
                cv.destroyAllWindows()
                break


cv.destroyAllWindows()
print(f"---{time.time() - start_time} secondi---")