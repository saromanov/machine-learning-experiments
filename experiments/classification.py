from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.datasets import load_digits


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