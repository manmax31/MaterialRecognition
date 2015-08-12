__author__ = 'manabchetia'

from os import listdir
from os.path import join

import cv2
import pandas as pd

MINC_DIR = '../../data/MINC/'
IMG_DIR = MINC_DIR + 'minc_orig'
TRAIN_FILE = MINC_DIR + 'minc/train.txt'
VALIDATE_FILE = MINC_DIR + 'minc/validate.txt'
TEST_FILE = MINC_DIR + 'minc/train.txt'


def get_img_files():
    '''
    This function indexes the names of each file in 10 different dataframes based on their extension
    :return:
    '''
    sub_dirs = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))
    dataFrames = []

    for dir in sub_dirs:
        imgs = filter(lambda x: x.endswith('.jpg'), listdir(join(IMG_DIR, dir)))
        df = pd.DataFrame(index=imgs, columns={'LABEL'})
        dataFrames.append(df)

    return dataFrames


if __name__ == '__main__':
    # dataFrames = get_img_files()
    details = '3,000000640,0.392972891373119,0.785225718194254'.split(',')
    img_name = details[1]
    x_comp = float(details[2])
    y_comp = float(details[3])

    img = IMG_DIR + '/0/' + img_name + '.jpg'
    image = cv2.imread(img)

    (h, w) = image.shape[:2]
    print 'Orginal: ', w, h
    x = w * x_comp
    y = h * y_comp
    print 'Patch Centers:', int(w), int(h)

    square_length = 0.233 * min(h, w)
    print 'Square\'s Length:', square_length

    add = int(square_length / 2)
    cropped = image[x - add: x + add, y - add: y + add]
    print cropped.shape[:2]
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    # cv2.imwrite("thumbnail.png", cropped)
