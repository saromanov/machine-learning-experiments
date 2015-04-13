import os
from getdatasets.get_german import get_german
import numpy as np
import sklearn

#https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

def load_dataset(path=None):
	""" Load german credit dataset """
	if path == None:
		get_german('./data')
	if os.path.exists(path):
		return open(path, 'rb').readlines()
	else:
		get_german(path)
		return open(path, 'rb').readlines()


def preprocessing(data):
	if data == None:
		raise Exception("This data is None")
	for value in data:
		pass
	array = []
	labels = []
	for value in data:
		pre = list(map(lambda x: int(x), value.split()))
		labels.append(pre[-1])
		array.append(pre[:-1])
	return np.array(array), np.array(labels)

def prediction(dataset):
	pass