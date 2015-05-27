from sklearn import manifold
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

iso = manifold.Isomap(10, n_components=2).fit_transform(X)
