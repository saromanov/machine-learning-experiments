from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

#Examples of classification text with sklearn

class NewsClassify:
    def __init__(self, method=RandomForestClassifier):
       self.train_data = fetch_20newsgroups(subset='train', categories=['comp.graphics', 'rec.sport.hockey',
        'talk.politics.misc', 'sci.med'])
       self.test_data = fetch_20newsgroups(subset='test', categories=['comp.graphics', 'rec.sport.hockey',
        'talk.politics.misc', 'sci.med'], shuffle=True, random_state=123)
       self.count_vec = CountVectorizer()
       CVTrain = self.count_vec.fit_transform(self.train_data.data)
       self.tfidf = TfidfTransformer()
       self.train =  self.tfidf.fit_transform(CVTrain)
       self.labels = self.train_data.target
       self.test = self.test_data.data
       self.model = method()

    def addDataset(self, train, labels, test):
       self.train = train
       self.labels = labels
       self.test = test

    def trainModel(self, method=RandomForestClassifier):
        self.model.fit(self.train, self.labels)

    def score(self):
        return np.mean(self.predict(self.test_data.data) == self.test_data.target)

    def predict(self, newdata):
        dock_new = self.count_vec.transform(newdata)
        X_new = self.tfidf.transform(dock_new)
        return self.model.predict(X_new)

model = NewsClassify()
model.trainModel()
print(model.score())
print(model.predict(['This team win six games at the raw']))