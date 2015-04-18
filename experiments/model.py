from abc import ABCMeta, abstractmethod

#Basic puppet for experiments

class Model(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def _preprocessing(self, X):
		pass

	@abstractmethod
	def _train(self, X, labels, method):
		pass

	@abstractmethod
	def predict(self, X):
		pass

	@abstractmethod
	def save(self, X):
		pass