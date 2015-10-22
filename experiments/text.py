from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.recurrent import JZS1
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.initializations import normal, identity

BATCH_SIZE = 32
EPOCHS = 20

RNN = JZS1
twenty_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'comp.graphics', 'sci.med'],
    shuffle=True, random_state=42)
counter = CountVectorizer()
train_counts = counter.fit_transform(twenty_train.data)
transform = TfidfTransformer(use_idf=False).fit_transform(train_counts)
model = Sequential()
model.add(RNN(10))
model.add(Activation('softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', class_mode='categorical')
model.fit(transform, twenty_train.target, batch_size=BATCH_SIZE, show_accuracy=True)
