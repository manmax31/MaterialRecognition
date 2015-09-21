__author__ = 'manabchetia'
'''
This script creates the train.txt, test.txt and val.txt required for LMDB creation
Contents of the files: [pathToImage classOfImage]
'''

from os import listdir
from os.path import join

MINC_DIR = '/srv/datasets/Materials/OpenSurfaces'
# MINC_DIR = '/Users/manabchetia/Documents/PyCharm/MaterialRecognition/data/MINC'
CATEGORIES_FILE = join(MINC_DIR, 'minc', 'categories.txt')
PATCH_DIR = join(MINC_DIR, 'patch')
TRAIN_BAL_DIR = join(PATCH_DIR, 'train_balance')
TEST_DIR = join(PATCH_DIR, 'test')
VAL_DIR = join(PATCH_DIR, 'val')
CATEGORIES = [line.strip() for line in open(CATEGORIES_FILE)]

if __name__ == '__main__':

    with open('test.txt', 'w') as test_file:
        categories_test = filter(lambda x: '.DS_Store' not in x, listdir(TEST_DIR))
        for category in categories_test:
            for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(TEST_DIR, category))):
                # print str(join(TEST_DIR, category, fil)), CATEGORIES.index(category)
                test_file.write(str(join(TEST_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')

    with open('val.txt', 'w') as val_file:
        categories_val = filter(lambda x: '.DS_Store' not in x, listdir(VAL_DIR))
        for category in categories_val:
            for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(VAL_DIR, category))):
                val_file.write(str(join(VAL_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')

    with open('train_balance.txt', 'w') as train_file:
        categories_train = filter(lambda x: '.DS_Store' not in x, listdir(TRAIN_BAL_DIR))
        for category in categories_train:
            for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(TRAIN_BAL_DIR, category))):
                train_file.write(str(join(TRAIN_BAL_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')
