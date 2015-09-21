__author__ = 'manabchetia'
'''
This script creates the train.txt, test.txt and val.txt required for LMDB creation
Contents of the files: [pathToImage classOfImage]
'''

from os import listdir
from os.path import join

SCALE = 256

DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/"
IMG_DIR = join(DTD_DIR, 'images')
OUTPUT_DIR = join(DTD_DIR, 'output')
ORIG_DIR = join(OUTPUT_DIR, 'images')
SCALED_DIR = join(OUTPUT_DIR, 'images_' + str(SCALE))
CATEGORIES = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))

if __name__ == '__main__':

    SETS = filter(lambda x: '.DS_Store' not in x, listdir(SCALED_DIR))

    for x in SETS:
        with open('test.txt', 'w') as test_file:
            for category in CATEGORIES:
                src_dir = join(SCALED_DIR, x, 'test', category)
                images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                label = str(CATEGORIES.index(category))
                for image in images:
                    test_file.write(src_dir + '/' + str(image) + ' ' + label + '\n')

                    # categories_test = filter(lambda x: '.DS_Store' not in x, listdir(TEST_DIR))
                    # for category in categories_test:
                    #     for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(TEST_DIR, category))):
                    #         # print str(join(TEST_DIR, category, fil)), CATEGORIES.index(category)
                    #         test_file.write(str(join(TEST_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')

                    # with open('val.txt', 'w') as val_file:
                    #     categories_val = filter(lambda x: '.DS_Store' not in x, listdir(VAL_DIR))
                    #     for category in categories_val:
                    #         for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(VAL_DIR, category))):
                    #             val_file.write(str(join(VAL_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')
                    #
                    # with open('train_balance.txt', 'w') as train_file:
                    #     categories_train = filter(lambda x: '.DS_Store' not in x, listdir(TRAIN_BAL_DIR))
                    #     for category in categories_train:
                    #         for fil in filter(lambda x: '.DS_Store' not in x, listdir(join(TRAIN_BAL_DIR, category))):
                    #             train_file.write(str(join(TRAIN_BAL_DIR, category, fil)) + ' ' + str(CATEGORIES.index(category)) + '\n')
