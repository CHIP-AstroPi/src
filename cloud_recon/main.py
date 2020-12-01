import os, sys
from pathlib import Path

from cloud_recon import recon


directory = os.fsencode("images")
print(directory)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        recon(filename)
        #print(filename)
    else:
        continue
