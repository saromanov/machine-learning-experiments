from getdatasets.get_fifteen_scene_categories import get_fifteen_scene_categories
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from model import Model
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import corner_harris
from skimage.filters import gaussian_filter
from skimage.transform import resize
import os

#Based on "Fifteen Scene Categories" dataset
#http://www-cvr.ai.uiuc.edu/ponce_grp/data/

''' Papers:
	Reconfigurable Models for Scene Recognition
	Learning Deep Features for Scene Recognition using Places Database
'''

class SceneClassification(Model):
	""" path is to dir with dataset """
	def __init__(self, path, method='rf'):
		self.dataset, self.labels = self._prepareDataset(path)
		self.trained_model = None
		self.method = method

	def _prepareDataset(self, path):
		labelnum = 0
		dataset = []
		labels = []
		for item in os.listdir(path)[4:10]:
			fullitem = path + item
			for dirpath, dirpaths, filenames in os.walk(fullitem):
				fullpaths = ['{0}/{1}'.format(dirpath, pathv) for pathv in filenames]
				for fullpath in fullpaths:
					value = resize(io.imread(fullpath), (32,32))
					dataset.append(corner_harris(gaussian_filter(value,0.1)))
					labels.append(labelnum)
					#print(corner_harris(value), value.shape)
				labelnum += 1
		print(len(labels))
		return np.array(dataset), np.array(labels, dtype=int)

	def get_score(self):
		""" 
		Show result of cost function after training 
		"""
		if self.trained_model == None:
			raise Exception("Model was not trained")
		return self.trained_model.score(self.test_data, self.test_labels)

	def _train(self):
		self.labels = self.labels.reshape(-1, 1)
		train_data, test_data, train_labels, test_labels = train_test_split(self.dataset, self.labels, test_size=0.2, random_state=124)
		train_data = np.reshape(train_data, (len(train_data),-1))
		test_data = np.reshape(test_data, (len(test_data),-1))
		self.train_data = train_data
		self.test_data = test_data
		self.train_labels = train_labels
		self.test_labels = test_labels
		classifier = RandomForestClassifier(max_depth=10, n_estimators=1000)
		classifier.fit(train_data, train_labels)
		self.trained_model = classifier
		return self.trained_model

	def predict(self, X):
		if self.trained_model == None:
			model = self._train()
			self.trained_model = model
			print(self.get_score())
		return self.trained_model.predict(X)

