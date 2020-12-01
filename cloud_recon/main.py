import numpy as np
import cv2, os, sys

from cloud_recon import recon


directory = os.fsencode("images")
print(directory)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(filename, 1)
        recon(filename, img)
    else:
        continue