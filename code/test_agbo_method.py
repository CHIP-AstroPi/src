import cv2 as cv
import numpy as np
from pathlib import Path, PurePath
from config import PATH_IMAGE, PATH_ELAB


"""
path questo è un semplice modo per gestire i file senza scrivere tutto il percorso ogni volta

credo due cartelle una dove il programma per i test prende le immagini e l'altra dove mette quelli elaborati

"""
Image_name = "image3.jpg"

cartella_Image: Path = Path(PATH_IMAGE)
cartella_elab : Path = Path(PATH_ELAB)

file_Image : PurePath= cartella_Image / Image_name


def nothing(x):
    pass

cv.namedWindow("Trackbars")
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:

    """
    per leggere l'immagine uso cv.imread(percorso) che ci permettere ogni singolo pixel dell'immagine
    sotto formato BGR (io preferivo usare Pillow)

    """
    Img = cv.imread(str(file_Image))

    #debug
    #print(Img)

    """
    uso cv.inRange perché devo dare un range di colori da prendere a opencv,cioè l'acqua.

    cv.inRange(image,low,upper)

    i colori in opencv sono memorizzati sotto il formato BGR blue-green-red e per memorizzarli uso 
    il metodo array di numpy (il formatto si può anche convertire )

    BGR = np.array([B,G,R])


    """

    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")

    #da BGR a RGB
    Img_HSV = cv.cvtColor(Img,cv.COLOR_BGR2RGB)

    low_HSV = np.array([l_h,l_s,l_v]) 

    upper_HSV = np.array([u_h,u_s,u_v]) 

    mask = cv.inRange(Img_HSV,low_HSV,upper_HSV)

    cv.imshow("frame",mask)
    

    key = cv.waitKey(1)
    if key == 27:
        break



cv.imwrite(f"{str(PATH_ELAB)}elab_{Image_name}",mask)
cv.destroyAllWindows()
