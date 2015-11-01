__author__ = 'manabchetia'

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image
from pprint import pprint
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
    #X_val = []
    
    y_train = np.repeat(np.linspace(0, len(CATEGORIES)-1, len(CATEGORIES)), 40)
    #y_train = []
    #y_test = []

    for label in labels:
        imgs_train = filter(lambda image: image.endswith('.jpg'), listdir(join(train_dir, label)))
        imgs_val   = filter(lambda image: image.endswith('.jpg'), listdir(join(val_dir, label)))
        #imgs_test  = filter(lambda image: image.endswith('.jpg'), listdir(join(test_dir, label)))

        imgs_train = map(lambda img: join(train_dir, label, img) , imgs_train)
        #imgs_val   = map(lambda img: join(val_dir, label, img) , imgs_val)
        #imgs_test  = map(lambda img: join(test_dir, label, img) , imgs_test)

        for img in imgs_train:
            img_arr = image.img_to_array(image.load_img(img).resize((224,224), Image.ANTIALIAS))
            X_train.append(img_arr)
        #for img in imgs_val:   X_val.append(image.img_to_array(image.load_img(img)))
        #for img in imgs_test:  X_test.append(image.img_to_array(image.load_img(img)))as

    return X_train, y_train



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
    #pprint(model.get_config())


    return model


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('/home/manab/caffe/examples/images/cat.jpg'), (224, 224))
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

   

    X_train, y_train = read_images(TRAIN_DIR, VAL_DIR, TEST_DIR)
    Y_train = np_utils.to_categorical(y_train, 47)


    datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=180,
    zca_whitening=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)



    #datagen.fit(np.asarray(X_train))

    

     # Test pretrained model
    model = VGG_16('/home/manab/Downloads/VGG_Keras_Model/vgg16_weights.h5')
    print("Model loaded ...")
    opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    print("Model compiled ...")
    out = model.predict(im)
    print CATEGORIES[np.argmax(out)]

    model.fit(X_train, y_train, nb_epoch=3, batch_size=40, verbose=2, show_accuracy=True)


    # nb_epoch = 2
    # for e in range(nb_epoch):
    #     print 'Epoch', e
    #     # batch train with realtime data augmentation
    #     for X_batch, Y_batch in datagen.flow(X_train, Y_train):
    #         loss = model.train(X_batch, Y_batch)


