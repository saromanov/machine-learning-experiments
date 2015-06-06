from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
#from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.datasets import imdb

max_features=20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(GRU(256, 128))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3, validation_split=0.1, show_accuracy=True)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, y_test)
print('Test accuracy:', acc)

