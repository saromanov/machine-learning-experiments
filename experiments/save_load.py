from sklearn.externals import joblib

def save(model, path):
	joblib.dump(model, path, compress=9)

def load(path):
	return joblib.load(path)