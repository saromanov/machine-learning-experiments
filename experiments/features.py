from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

#Experiments with deature importance and feature selection

def iris_sample():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)