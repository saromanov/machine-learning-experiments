from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.datasets.data_utils import get_file
from six.moves import range


def stacked_lstm(iters=30, diversity=[0.1,0.5,1.5]):
    maxlen = 20
    text = open('text').read().lower()
    chars = set(text)
    model = Sequential()
    model.add(LSTM(256,256))
    model.add(Dropout(0.3))
    model.add(LSTM(128,128))
    model.add(Dropout(0.3))
    model.add(LSTM(128,128))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    for i in range(iters):
        model.fit(X, y, batch_size=128,, nb_epoch=1)
        for diver in diversity:
            print('Diversity: {0}'.format(diver))
            generated = ''
            sent = text[start:start + maxlen]
            generated + sent
            print(generated)
            for it in range(200):
                pass


def conv():
    batch_size = 32
    nb_classes = 10
    nb_epoch = 200
    data_augmentation = True
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64*8*8, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

conv()
