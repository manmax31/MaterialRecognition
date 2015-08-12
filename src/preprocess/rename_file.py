__author__ = 'manabchetia'

folder = "2"
MINC_DIR = '../../data/MINC/'
IMG_DIR = MINC_DIR + 'minc_orig/' + folder

import os

for filename in os.listdir(IMG_DIR):
    if filename.endswith(".jpg"):
        name_ext = filename.split(".")
        name = name_ext[0]
        ext = name_ext[1]

        os.rename(IMG_DIR + "/" + filename, IMG_DIR + "/" + name[:-1] + folder + '.jpg')
