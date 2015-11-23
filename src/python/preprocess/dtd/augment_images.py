__author__ = 'manabchetia'
'''
This script augments data by random rotation
'''

from os import listdir, makedirs
from os.path import join, exists
from PIL import Image
import random


import numpy as np
import cv2

SCALE = '384'

DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/"
IMG_DIR = join(DTD_DIR, 'images')
OUTPUT_DIR = join(DTD_DIR, 'output')
ORIG_DIR = join(OUTPUT_DIR, 'images')
ROTATE_DIR = join(OUTPUT_DIR, 'images_' + str(SCALE))
CATEGORIES = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))


def prepare_folders():
    """
    This function creates all necessary folders
    :return:
    """
    # creates the 10 folders as there train1, ...,train10, test1,...test10 and val1,...val10 files
    map(lambda x: makedirs(join(ROTATE_DIR, str(x))) if not exists(join(ROTATE_DIR, str(x))) else None, xrange(1, 11))

    # Creates folders train,test,validate and class folders inside them
    SETS = filter(lambda x: '.DS_Store' not in x, listdir(ROTATE_DIR))
    for x in SETS:
        for group in ['train', 'test', 'validate']:
            for category in CATEGORIES:
                dir = join(ROTATE_DIR, x, group, category)
                if not exists(dir):
                    makedirs(dir)


def rotate_image(img_name, src_path, dst_path):
    """
    This function randomly rotates image between 0 and 180 degrees
    :param img_name: string containing the name of the image
    :param src_path: path of the image from image is read
    :param dst_path: path of the image where is to be written
    :return:
    """

    if random.random() <= 0.5:
        img = Image.open(join(src_path, img_name))
        img = img.rotate(random.randint(0,180))
        img.save(join(dst_path, img_name[:-4]+'-rotated.jpg'), format="JPEG", subsampling=0, quality=100)


if __name__ == '__main__':
    # prepare_folders()

    # SETS = filter(lambda x: '.DS_Store' not in x, listdir(ORIG_DIR))
    # for x in SETS:
    #     print 'Set', x, 'is being processed...'
    #     for group in ['train', 'test', 'validate']:
    #         for category in CATEGORIES:
                # src_dir = join(ORIG_DIR, x, group, category)
                # dst_dir = join(ROTATE_DIR, x, group, category)
                # images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                # # map(lambda image: rotate_image(image, src_dir, dst_dir), images)
                # for image in images:
                #     rotate_image(image, src_dir, dst_dir)
 
    x = str(4)
    group = 'train'

    for category in CATEGORIES:
        src_dir = join(ORIG_DIR, x, group, category)
        dst_dir = join(ROTATE_DIR, x, group, category)
        images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                # map(lambda image: rotate_image(image, src_dir, dst_dir), images)
        for image in images:
            rotate_image(image, src_dir, dst_dir)
