import os, sys

from cloud_recon import recon

for file in os.listdir("images"):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        recon(filename)
    else:
        continue