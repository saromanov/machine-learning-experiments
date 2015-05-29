from sklearn import manifold, svm
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split


def isomap(X, components):
    return manifold.Isomap(10, n_components=components).fit_transform(X)


def tsne1(X, components):
    return manifold.TSNE(n_components=components, learning_rate=100, n_iter=300).fit_transform(X)


def tsne2(X, components):
    return manifold.TSNE(n_component=components, learning_rate=1000, n_iter=400).fit_transform(X)


def mds1(X, components):
    return manifold.MDS(n_components=components).fit_transform(X)


def create_model(X, y, msg):
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=124)
    model = svm.SVC(gamma=0.001)
    model.fit(train_data, train_labels)
    print("{0} {1}".format(msg, model.score(test_data, test_labels)))


digits = load_digits()
X = digits.data
y = digits.target
newX = isomap(X, 50)
create_model(newX, y, "With Isomap and 50 comonents")
newX2 = isomap(X, 20)
create_model(newX2, y, "With Isomap and 20 comonents")
newX3 = mds1(X, 20)
create_model(newX3, y, "With MDS and 20 components")
create_model(X, y, "Simple model")
