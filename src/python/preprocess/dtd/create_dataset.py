__author__ = 'manabchetia'
'''
This script creates the DTD dataset based on train.txt, test.txt and validate.txt
'''
from os import listdir, makedirs
from os.path import join, exists
import shutil
import errno
from pprint import pprint

# DTD_DIR = "/Users/manabchetia/Documents/PyCharm/MaterialRecognition/data/dtd"
DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd"
IMG_DIR = join(DTD_DIR, 'images')
FILE_DIR = join(DTD_DIR, 'labels')
OUTPUT_DIR = join(DTD_DIR, 'output')
CATEGORIES = filter(lambda x: '.DS_Store' not in x, listdir(IMG_DIR))


def prepare_folders():
    """
    This function creates all necessary folders
    :return:
    """
    # creates the 10 folders as there train1, ...,train10, test1,...test10 and val1,...val10 files
    map(lambda x: makedirs(join(OUTPUT_DIR, str(x))) if not exists(join(OUTPUT_DIR, str(x))) else None, xrange(1, 11))

    # Creates folders train,test,validate and class folders inside them
    SETS = filter(lambda x: '.DS_Store' not in x, listdir(OUTPUT_DIR))
    for x in SETS:
        for group in ['train', 'test', 'validate']:
            for category in CATEGORIES:
                dir = join(OUTPUT_DIR, x, group, category)
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
    pprint(CATEGORIES)
    # prepare_folders()

    # for SET in xrange(1, 11):
    #     TRAIN_FILE = join(FILE_DIR, 'train' + str(SET) + '.txt')
    #     TEST_FILE = join(FILE_DIR, 'test' + str(SET) + '.txt')
    #     VAL_FILE = join(FILE_DIR, 'val' + str(SET) + '.txt')

    #     with open(TRAIN_FILE) as train_file:
    #         for line in train_file:
    #             line = line.strip().split('/')
    #             category = line[0]
    #             img_name = line[1]
    #             src = join(IMG_DIR, category)
    #             dst = join(OUTPUT_DIR, str(SET), 'train', category)
    #             copy(join(src, img_name), join(dst, img_name))

    #     with open(TEST_FILE) as test_file:
    #         for line in test_file:
    #             line = line.strip().split('/')
    #             category = line[0]
    #             img_name = line[1]
    #             src = join(IMG_DIR, category)
    #             dst = join(OUTPUT_DIR, str(SET), 'test', category)
    #             copy(join(src, img_name), join(dst, img_name))

    #     with open(VAL_FILE) as val_file:
    #         for line in val_file:
    #             line = line.strip().split('/')
    #             category = line[0]
    #             img_name = line[1]
    #             src = join(IMG_DIR, category)
    #             dst = join(OUTPUT_DIR, str(SET), 'validate', category)
    #             copy(join(src, img_name), join(dst, img_name))
