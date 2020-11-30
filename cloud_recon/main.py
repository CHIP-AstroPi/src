import os, sys

from cloud_recon import recon

directory = os.fsencode("images")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        recon(filename)
    else:
        continue