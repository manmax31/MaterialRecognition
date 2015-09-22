__author__ = 'manabchetia'
'''
This script creates a new dataset based on SCALE = 256 or 384 or [256:384]
'''

from os import listdir, makedirs
from os.path import join, exists
import random

import numpy as np
import cv2

SCALE = '256_512'

DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/"
IMG_DIR = join(DTD_DIR, 'images')
OUTPUT_DIR = join(DTD_DIR, 'output')
ORIG_DIR = join(OUTPUT_DIR, 'images')
SCALED_DIR = join(OUTPUT_DIR, 'images_' + str(SCALE))
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


def scale_image(img_name, src_path, dst_path):
    """
    This function resizes the smallest dimension of the image to the SCALE and maintains the aspect ratio
    :param img_name: string containing the name of the image
    :param src_path: path of the image from image is read
    :param dst_path: path of the image where is to be written
    :return:
    """
    img = cv2.imread(join(src_path, img_name))
    height, width, _ = img.shape
    min_dim = np.argmin([width, height])

    # global SCALE
    if SCALE == '256_512':
        scale = random.randint(256, 512)

    if min_dim == 0:
        new_width = scale  # SCALE
        new_height = new_width * height / width
    else:
        new_height = scale  # SCALE
        new_width = new_height * width / height

    # img = img.resize((new_width, new_width), Image.ANTIALIAS)
    # img.save
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(join(dst_path, img_name), img)


if __name__ == '__main__':
    prepare_folders()

    SETS = filter(lambda x: '.DS_Store' not in x, listdir(ORIG_DIR))
    for x in SETS:
        print 'Set', x, 'is being processed...'
        for group in ['train', 'test', 'validate']:
            for category in CATEGORIES:
                src_dir = join(ORIG_DIR, x, group, category)
                dst_dir = join(SCALED_DIR, x, group, category)
                images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                # map(lambda image: scale_image(image, src_dir, dst_dir), images)
                for image in images:
                    scale_image(image, src_dir, dst_dir)
