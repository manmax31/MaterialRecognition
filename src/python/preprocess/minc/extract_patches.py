__author__ = 'manabchetia'
'''
This script extracts patches from images from train.txt, categories.txt and validate.txt
'''

from os import listdir, makedirs
from os.path import join, exists
import shutil
import errno

import cv2
import pandas as pd

MINC_DIR = '../../../data/MINC/'
CATEGORIES_FILE = join(MINC_DIR, 'minc', 'categories.txt')
IMG_DIR = join(MINC_DIR, 'minc_orig')
TRAIN_FILE = join(MINC_DIR, 'minc', 'train.txt')
VALIDATE_FILE = join(MINC_DIR, 'minc', 'validate.txt')
TEST_FILE = join(MINC_DIR, 'minc', 'train.txt')
PATCH_DIR = join(MINC_DIR, 'patch')
CATEGORIES = [line.strip() for line in open(CATEGORIES_FILE)]
N_CLASSES = len(CATEGORIES)


def get_img_files():
    '''
    This function indexes the names of each file in 10 different dataframes based on their extension
    :return: dataFrames
    '''
    sub_dirs = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))
    dataFrames = []

    for dir in sub_dirs:
        imgs = filter(lambda x: x.endswith('.jpg'), listdir(join(IMG_DIR, dir)))
        df = pd.DataFrame(index=imgs, columns={'LABEL'})
        dataFrames.append(df)

    return dataFrames


def extract_patch(label, filename, patch_center_x_norm, patch_center_y_norm, group, img_counter):
    '''
    This function extracts patches from images and writes those patches
    :param label: class of the image
    :param filename: name of the file
    :param patch_center_x_norm: [0-1] file width
    :param patch_center_y_norm: [0-1] file height
    :param group: folder 'train', 'validate' or 'test'
    :return: nil
    '''
    img = join(IMG_DIR, filename[-1], filename + '.jpg')
    if not exists(img):
        pass
    else:
        image = cv2.imread(img)

        (h, w) = image.shape[:2]

        patch_center_x = w * patch_center_x_norm
        patch_center_y = h * patch_center_y_norm
        patch_length = 0.233 * min(h, w)

        add = int(patch_length / 2)
        y_l = patch_center_y - add
        if y_l < 0:
            y_l = 1
        y_r = patch_center_y + add
        x_l = patch_center_x - add
        if x_l < 0:
            x_l = 1
        x_r = patch_center_x + add

        patch = image[y_l:y_r, x_l:x_r]
        cv2.imwrite(join(PATCH_DIR, group, CATEGORIES[int(label)], str(img_counter) + '.jpg'), patch)


def create_patch_class_dirs():
    '''
    This function creates directories of classes [0, 1, 2, ..., 22] inside train or validate or test
    :return: nil
    '''

    for group in ['train', 'test', 'validate']:
        for category in CATEGORIES:
            dir = join(PATCH_DIR, group, category)
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


if __name__ == '__main__':
    # create_patch_class_dirs()
    #
    img_counter = 1
    with open(VALIDATE_FILE) as val_file:
        for line in val_file:
            details = line.strip().split(',')
            extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'validate', img_counter)
            img_counter += 1
            #
            # img_counter = 1
            # with open(TEST_FILE) as test_file:
            #     for line in test_file:
            #         details = line.strip().split(',')
            #         extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'test', img_counter)
            #         img_counter += 1
            #
            # img_counter = 1
            # with open(TRAIN_FILE) as train_file:
            #     for line in train_file:
            #         details = line.strip().split(',')
            #         extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'train', img_counter)
            #         img_counter += 1
