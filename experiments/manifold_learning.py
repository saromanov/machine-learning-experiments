from sklearn import manifold
from sklearn.datasets import load_digits

def isomap(X):
    return manifold.Isomap(10, n_components=20).fit_transform(X)

def tsne1(X, components):
    return manifold.TSNE(n_components=components, learning_rate=100, n_iter=300).fit_transform(X)

def tsne2(X, components):
    return manifold.TSNE(n_components_components, learning_rate=1000, n_iter=400).fit_transform(X)

digits = load_digits()
X = digits.data
#tsne1(X,2)
