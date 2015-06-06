from sklearn.externals import joblib

def save(path):
	joblib.dump(newX, path, compress=9)

def load(path):
	return joblib.load(path)