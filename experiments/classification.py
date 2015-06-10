from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

#Eathquake classification

def ABC(Xtrain, ytrain, Xtest):
	model = AdaBoostClassifier(n_estimators=100, learning_rate=0.001)
	return model.fit(Xtrain,ytrain)

def GBC(Xtrain, ytrain, Xtest):
	model = GradientBoostingClassifier(max_depth=5, learning_rate=0.001)
	return model.fit(Xtrain, ytrain)

def BC(Xtrain, ytrain, Xtest):
	model = BaggingClassifier(bootstrap=True)
	return model.fit(Xtrain, ytrain)

def predict(model, Xtest, ytest):
    return model.predict(Xtest), model.score(Xtest, ytest)

def global_earthquake():
    digits = fetch_mldata('global-earthquakes')
    X = digits.data
    y = digits.target
    train_data, test_data, train_labels, test_labels = train_test_split(X,  y[1], test_size=0.2, random_state=124)
    model = BC(train_data, train_labels, test_data)
    result, score = predict(model, test_data, test_labels)

def mnist():
    basket = fetch_mldata('mnist')
    X = basket.data
    y = basket.target
    print(X)
    train_data, test_data, train_labels, test_labels = train_test_split(X,  y, test_size=0.2, random_state=124)
    model = BC(train_data, train_labels, test_data)
    result, score = predict(model, test_data, test_labels)
    print(score)

def ACASVA_actions_dataset():
    action = fetch_mldata('')


mnist()