from sklearn import manifold, svm
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from save_load import save

def isomap(X, components):
    return manifold.Isomap(10, n_components=components).fit_transform(X)


def tsne1(X, components):
    return manifold.TSNE(n_components=components, random_state=1234).fit_transform(X)


def tsne2(X, components):
    return manifold.TSNE(n_components=components, learning_rate=1000, n_iter=400).fit_transform(X)


def mds1(X, components):
    return manifold.MDS(n_components=components).fit_transform(X)

def se(X, components):
	return manifold.SpectralEmbedding(n_components=components).fit_transform(X)


def create_model(X, y, msg):
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=124)
    model = svm.SVC(gamma=0.001)
    model.fit(train_data, train_labels)
    print("{0} {1}".format(msg, model.score(test_data, test_labels)))

def create_model2(X, y, msg):
	pass


def getAndSave():
	digits = load_digits()
	X = digits.data
	y = digits.target
	newX = isomap(X, 50)
	save(newX, "./models/isomap50.pkl")
	create_model(newX, y, "With Isomap and 50 comonents")
	newX2 = isomap(X, 20)
	save(newX2, "./models/isomap20.pkl")
	create_model(newX2, y, "With Isomap and 20 comonents")
	newX3 = mds1(X, 20)
	save(newX3, "./models/mds20.pkl")
	create_model(newX3, y, "With MDS and 20 components")
	newX4 = tsne1(X,20)
	create_model(newX4, y, "With t-SNE and 20 components")
	newX5 = se(X, 2)
	create_model(newX5, y, "With Spectral Embedding and 5 components")
	create_model(X, y, "SVM, but without dimensionality reduction")

getAndSave()
