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
        with open(join(SCALED_DIR, x, 'test.txt'), 'w') as test_file:
            for category in CATEGORIES:
                src_dir = join(SCALED_DIR, x, 'test', category)
                images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                label = str(CATEGORIES.index(category))
                for image in images:
                    test_file.write(src_dir + '/' + str(image) + ' ' + label + '\n')

        with open(join(SCALED_DIR, x, 'train.txt'), 'w') as train_file:
            for category in CATEGORIES:
                src_dir = join(SCALED_DIR, x, 'train', category)
                images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                label = str(CATEGORIES.index(category))
                for image in images:
                    train_file.write(src_dir + '/' + str(image) + ' ' + label + '\n')

        with open(join(SCALED_DIR, x, 'validate.txt'), 'w') as val_file:
            for category in CATEGORIES:
                src_dir = join(SCALED_DIR, x, 'validate', category)
                images = filter(lambda image: image.endswith('.jpg'), listdir(src_dir))
                label = str(CATEGORIES.index(category))
                for image in images:
                    val_file.write(src_dir + '/' + str(image) + ' ' + label + '\n')
