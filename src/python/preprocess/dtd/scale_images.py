__author__ = 'manabchetia'

from os import listdir, makedirs
from os.path import join, exists

import numpy as np
import cv2

SCALE = 256

DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/"
IMG_DIR = join(DTD_DIR, 'images')
OUTPUT_DIR = join(DTD_DIR, 'output')
ORIG_DIR = join(OUTPUT_DIR, 'images')
SCALED_DIR = join(OUTPUT_DIR, 'image_' + str(SCALE))
CATEGORIES = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))


def prepare_folders():
    """
    This function creates all necessary folders
    :return:
    """
    # creates the 10 folders as there train1, ...,train10, test1,...test10 and val1,...val10 files
    map(lambda x: makedirs(join(SCALED_DIR, str(x))) if not exists(join(SCALED_DIR, str(x))) else None, xrange(1, 11))

    # Creates folders train,test,validate and class folders inside them
    SETS = filter(lambda x: '.DS_Store' not in x, listdir(SCALED_DIR))
    for x in SETS:
        for group in ['train', 'test', 'validate']:
            for category in CATEGORIES:
                dir = join(SCALED_DIR, x, group, category)
                if not exists(dir):
                    makedirs(dir)


def scale_image(img_name, path):
    """
    This function resizes the smallest dimension of the image to the SCALE and maintains the aspect ratio
    :param img_name: string containing the name of the image
    :param path: path of the image where is to be written
    :return:
    """
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    min_dim = np.argmin([width, height])

    if min_dim == 0:
        new_width = SCALE
        new_height = new_width * height / width
    else:
        new_height = SCALE
        new_width = new_height * width / height

    # img = img.resize((new_width, new_width), Image.ANTIALIAS)
    # img.save
    img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(path, img)


if __name__ == '__main__':
    prepare_folders()

    SETS = filter(lambda x: '.DS_Store' not in x, listdir(ORIG_DIR))
    for x in SETS:
        for group in ['train', 'test', 'validate']:
            for category in CATEGORIES:
                images = filter(lambda image: image.endswith('.jpg'), listdir(join(ORIG_DIR, x, group, category)))
                print images
