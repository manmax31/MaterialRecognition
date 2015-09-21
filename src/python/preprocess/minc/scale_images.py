__author__ = 'manabchetia'

from os import listdir, makedirs
from os.path import join, exists
import shutil
import errno

from scipy.misc import imresize, imsave, imread
import numpy as np


# MINC_DIR    = '/srv/datasets/Materials/OpenSurfaces'
MINC_DIR = '/Users/manabchetia/Documents/PyCharm/MaterialRecognition/data/MINC'
PATCH_DIR = join(MINC_DIR, 'patch')
TEST_DIR = join(PATCH_DIR, 'test')
VAL_DIR = join(PATCH_DIR, 'val')
TEST_SCALED_DIR = join(PATCH_DIR, 'test_scaled')
VAL_SCALED_DIR = join(PATCH_DIR, 'val_scaled')
CATEGORIES_FILE = join(MINC_DIR, 'minc', 'categories.txt')
CATEGORIES = [line.strip() for line in open(CATEGORIES_FILE)]


def create_class_dirs(group):
    for category in CATEGORIES:
        dir = join(group, category)
        if not exists(dir):
            makedirs(dir)


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


def scale_images(source, destination):
    for category in CATEGORIES:
        src_dir = join(PATCH_DIR, source, category)
        imgs = filter(lambda x: x.endswith('.jpg'), listdir(src_dir))
        for img in imgs:
            copy(join(src_dir, img), join(destination, category, img))
            image = imread(join(src_dir, img))
            image_small = imresize(image, 1.0 / np.sqrt(2))
            image_big = imresize(image, np.sqrt(2))
            imsave(join(destination, category, img.split('.')[0] + '_small.jpg'), image_small)
            imsave(join(destination, category, img.split('.')[0] + '_big.jpg'), image_big)
        print(imgs)


if __name__ == '__main__':
    print filter(lambda x: '.DS_Store' not in x, listdir(TEST_DIR))
    print CATEGORIES
    create_class_dirs(TEST_SCALED_DIR)
    create_class_dirs(VAL_SCALED_DIR)

    scale_images(TEST_DIR, TEST_SCALED_DIR)
