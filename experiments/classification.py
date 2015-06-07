from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


def ABC(Xtrain, ytrain, Xtest):
	model = AdaBoostClassifier(n_estimators=100, learning_rate=0.001)
	model.fit(Xtrain,ytrain)
	return model.predict(Xtest)

def GBC(Xtrain, ytrain, Xtest):
	model = GradientBoostingClassifier(max_depth=5, learning_rate=0.001)
	model.fit(Xtrain, ytrain)
	return model.predict(Xtest)

def BC(Xtrain, ytrain, Xtest):
	model = BaggingClassifier(bootsrap=True)
	model.fit(Xtrain, ytrain)
	return model.predict(Xtest)

digits = fetch_mldata('global-earthquakes')
X = digits.data
y = digits.target
train_data, test_data, train_labels, test_labels = train_test_split(X,  y[1], test_size=0.2, random_state=124)
GBC(train_data, train_labels, test_labels)