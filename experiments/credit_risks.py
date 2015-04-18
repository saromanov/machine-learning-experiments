import os
from getdatasets.get_german import get_german
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from model import Model
import pickle

#https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

class CreditRisks(Model):
	def __init__(self, path=None):
		self.data = None
		self.labels = None
		if path != None:
			self.data, self.labels = self._prepocessing(self.load_dataset(path=path))
		self.trained_model = None

	def set_training_params(self, data, labels):
		self.data = data
		self.labels = labels

	def load_dataset(self, path=None):
		""" Load german credit dataset """
		if path == None:
			get_german('./data')
		if os.path.exists(path):
			return open(path, 'rb').readlines()
		else:
			get_german(path)
			return open(path, 'rb').readlines()

	def _preprocessing_prediction(self, path):
		data = open(path, 'rb').readlines()
		result = []
		return [line.split() for line in data]

	def _prepocessing(self, data):
		if data == None:
			raise Exception("dataset not contain any elements")
		array = []
		labels = []
		for value in data:
			pre = list(map(lambda x: int(x), value.split()))
			labels.append(pre[-1])
			array.append(pre[:-1])
		return np.array(array), np.array(labels)

	def get_score(self):
		""" 
		Show result of cost function after training 
		"""
		if self.trained_model == None:
			raise Exception("Model was not trained")
		return self.trained_model.score(self.test_data, self.test_labels)

	def _train(self):
		train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=124)
		self.train_data = train_data
		self.test_data = test_data
		self.train_labels = train_labels
		self.test_labels = test_labels
		low_dim_input = PCA().fit_transform(train_data)
		svmfit = svm.SVC(gamma=0.001, kernel='linear').fit(train_data, train_labels)
		new_labels = svmfit.predict(test_data)
		return new_labels, svmfit

	def predict(self, newdata):
		if self.trained_model == None:
			newlabels, model = self._train()
			self.trained_model = model
		return self.trained_model.predict(newdata)

	def save(self, outfile):
		""" http://scikit-learn.org/stable/modules/model_persistence.html
		"""
		if self.trained_model != None:
			joblib.dump(self.trained_model, outfile) 

	
