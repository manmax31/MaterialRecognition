__author__ = 'manabchetia'
'''
This script creates the balanced dataset as described in the paper (UNDERSAMPLING)
'''

from os import listdir, makedirs
from os.path import join, exists
import random
import shutil
import errno

import numpy as np

MINC_DIR = '../../../data/MINC/'
CATEGORIES_FILE = join(MINC_DIR, 'minc', 'categories.txt')
PATCH_DIR = join(MINC_DIR, 'patch')
PATCH_BAL_DIR = join(MINC_DIR, 'patch-balance-under')
TRAIN_DIR = join(PATCH_DIR, 'train')
VAL_DIR = join(PATCH_DIR, 'validate')
TEST_DIR = join(PATCH_DIR, 'test')
CATEGORIES = [line.strip() for line in open(CATEGORIES_FILE)]
N_CLASSES = len(CATEGORIES)


def create_patch_class_dirs():
    """
    This function creates directories of classes [0, 1, 2, ..., 22] inside train or validate or test
    :return: nil
    """

    for group in ['train', 'test', 'validate']:
        for category in CATEGORIES:
            dir = join(PATCH_BAL_DIR, group, category)
            if not exists(dir):
                makedirs(dir)


def get_img_files(group):
    """
    This function indexes the names of each file in 10 different dataframes based on their extension
    :return: list of list of images in each class [ ['1.jpg'], ['2.jpg', '3.jpg'], [''5.jpg''], ...]
    """
    return [filter(lambda x: x.endswith('.jpg'), listdir(join(PATCH_DIR, group, dir))) for dir in CATEGORIES]


def get_smallest_category(img_list):
    """
    This function gets the class with minimum images and the number of images in it
    :param img_list:
    :return min_class: class_name with minimum images. e.g. 'wood'
    :return min_qty: number of images in minimum class
    """
    qty_list = map(len, get_img_files('validate'))
    min_qty = min(qty_list)
    min_class = CATEGORIES[np.argmin(qty_list)]

    return min_class, min_qty


def copy(src, dst):
    """
    This function copies files from src to dst
    :param src: source file
    :param dst: destination file
    :return:
    """
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def run_copy(group):
    """
    This function does the actual copy
    :param group: 'train', 'test' or 'validate'
    :return:
    """
    imgs_list = get_img_files(group)
    min_category, min_qty = get_smallest_category(imgs_list)

    for dir in CATEGORIES:
        src_dir = join(PATCH_DIR, group, dir)
        dst_dir = join(PATCH_BAL_DIR, group, dir)

        imgs = imgs_list[CATEGORIES.index(dir)]
        n_imgs = len(imgs)

        random_img_indices = random.sample(xrange(0, n_imgs), min_qty)

        for index in random_img_indices:
            img = imgs[index]
            copy(join(src_dir, img), join(dst_dir, img))


if __name__ == '__main__':
    create_patch_class_dirs()
    run_copy('validate')
    run_copy('train')
    run_copy('test')
