__author__ = 'manabchetia'

from os import listdir, makedirs
from os.path import join, exists

import cv2
import pandas as pd

MINC_DIR = '../../data/MINC/'
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

        ## Manab 1
        (h, w) = image.shape[:2]

        patch_center_x = w * patch_center_x_norm
        patch_center_y = h * patch_center_y_norm
        patch_length = 0.233 * min(h, w)

        add = int(patch_length / 2)
        y_l = patch_center_y - add
        y_r = patch_center_y + add
        x_l = patch_center_x - add
        x_r = patch_center_x + add
        if y_l > 0 and y_r > 0 and x_l > 0 and x_r > 0:
            patch = image[y_l:y_r, x_l:x_r]
            cv2.imwrite(join(PATCH_DIR, group, CATEGORIES[int(label)], str(img_counter) + '.jpg'), patch)
        else:
            pass

            # patch = image[patch_center_y - add : patch_center_y + add, patch_center_x - add : patch_center_x + add]
            ## Manab 1

            # cv2.imwrite(join(PATCH_DIR, group, CATEGORIES[int(label)], str(img_counter)+'.jpg'), patch)


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

if __name__ == '__main__':
    # dataFrames = get_img_files()
    # details = '3,000000640,0.392972891373119,0.785225718194254'.split(',')

    create_patch_class_dirs()

    img_counter = 1
    with open(VALIDATE_FILE) as val_file:
        for line in val_file:
            details = line.strip().split(',')
            extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'validate', img_counter)
            img_counter += 1

            # img_counter = 1
            # with open(TEST_FILE) as test_file:
            #     for line in test_file:
            #         details = line.strip().split(',')
            #         extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'test', img_counter)
            # #

            # img_counter = 1
            # with open(TRAIN_FILE) as train_file:
            #     for line in train_file:
            #         details = line.strip().split(',')
            #         extract_patch(details[0], details[1], float(details[2]), float(details[3]), 'train', img_counter)







            # img_name = details[1]
            # x_comp = float(details[2])
            # y_comp = float(details[3])
            # #
            # img = IMG_DIR + '/0/' + img_name + '.jpg'
            # image = cv2.imread(img)
            # #
            # (h, w) = image.shape[:2]
            # print 'Orginal: ', w, h
            # x = w * x_comp
            # y = h * y_comp
            # # print 'Patch Centers:', int(w), int(h)
            #
            # patch_length = 0.233 * min(h, w)
            # print 'Square\'s Length:', patch_length
            # #
            # add = int(patch_length / 2)
            # cropped = image[x - add: x + add, y - add: y + add]
            # print cropped.shape[:2]
            # cv2.imshow("cropped", cropped)
            # cv2.waitKey(0)
            # # # cv2.imwrite("thumbnail.png", cropped)
            #
            # # image = cv2.imread(img)
            # # patch_center = np.array([x, y])
            # # smaller_dim = np.min(image.shape[0:2])
            # # patch_scale = 0.233
            # # patch_size = patch_scale * smaller_dim
            # # patch_x = patch_center[0] - patch_size/2
            # # patch_y = patch_center[1] - patch_size/2
            # # patch_image = image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
            # # cv2.imshow("cropped", patch_image)
            # # cv2.waitKey(0)
