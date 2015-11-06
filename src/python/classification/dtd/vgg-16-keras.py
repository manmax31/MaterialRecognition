__author__ = 'manabchetia'

from os import listdir
from os.path import join

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from PIL import Image
import numpy as np

import cv2

SCALE = 384
SET = 3
DTD_DIR = "/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/"
SCALE_DIR = join(DTD_DIR, 'output','images_' + str(SCALE))
SET_DIR = join(SCALE_DIR, str(SET))
TRAIN_DIR = join(SET_DIR, 'train')
VAL_DIR = join(SET_DIR, 'validate')
TEST_DIR = join(SET_DIR, 'test')
CATEGORIES = filter(lambda x: '.DS_Store' not in x, listdir(TRAIN_DIR))


def read_images(train_dir, val_dir, test_dir):
    labels = filter(lambda x: '.DS_Store' not in x, listdir(train_dir))
    
    X_train = []
    #X_test = []
    X_val = []
    
    y_train = np.repeat(np.linspace(0, len(CATEGORIES)-1, len(CATEGORIES)), 40)
    y_val = []
    #y_test = []

    for label in labels:
        imgs_train = filter(lambda image: image.endswith('.jpg'), listdir(join(train_dir, label)))
        imgs_val   = filter(lambda image: image.endswith('.jpg'), listdir(join(val_dir, label)))
        #imgs_test  = filter(lambda image: image.endswith('.jpg'), listdir(join(test_dir, label)))

        imgs_train = map(lambda img: join(train_dir, label, img) , imgs_train)
        imgs_val   = map(lambda img: join(val_dir, label, img) , imgs_val)
        #imgs_test  = map(lambda img: join(test_dir, label, img) , imgs_test)

        for img in imgs_train:
            img_arr = image.img_to_array(image.load_img(img).resize((224,224), Image.ANTIALIAS))
            X_train.append(img_arr)
        for img in imgs_val:
            img_arr = image.img_to_array(image.load_img(img).resize((224,224), Image.ANTIALIAS))
            X_val.append(img_arr)
        #for img in imgs_test:  X_test.append(image.img_to_array(image.load_img(img)))as

    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)



def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512 * 7 * 7, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    
    #pprint(model.layers)
    model.layers.pop()
    model.params.pop()
    model.layers.pop()
    model.params.pop()


    model.add(Dense(4096, 47, activation='softmax'))
    # print(model.get_config())
    # model.layers.pop()
    return model


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = read_images(TRAIN_DIR, VAL_DIR, TEST_DIR)
    Y_train = np_utils.to_categorical(y_train, 47)
    Y_test = np_utils.to_categorical(y_test, 47)

    datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=180,
    zca_whitening=False,
    horizontal_flip=True,
    vertical_flip=True)

    datagen.fit(X_train)

    #  # Test pretrained model
    model = VGG_16('/home/manab/Downloads/VGG_Keras_Model/vgg16_weights.h5')
    print("Model loaded ...")
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    print("Model compiled ...")


    # model.fit(X_train, y_train, nb_epoch=3, batch_size=40, verbose=2, show_accuracy=True)

    nb_epoch = 100
    for e in xrange(nb_epoch):
        print('\nEpoch '+str(e))
        # print("Training...")
        # progbar_train = generic_utils.Progbar(X_train.shape[0])
        # for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=66):
        #     (loss, acc) = model.train_on_batch(X_batch, Y_batch, accuracy=True)
        #     progbar_train.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar_test = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            (loss, acc) = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar_test.add(X_batch.shape[0], values=[('test acc', acc)])


