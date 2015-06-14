from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree, cross_validation, decomposition
from sklearn.datasets import fetch_mldata, load_iris
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


def classify_mnist():
    digits = fetch_mldata('MNIST Original')
    X = digits.data
    y = digits.target
    print(X.shape)
    train_data, test_data, train_labels, test_labels = train_test_split(X,  y, test_size=0.2, random_state=124)
    model = Pipeline([('pca', decomposition.PCA()), ('feature_selection', LogisticRegression()), ('classification', SVC(kernel='linear'))])
    model.fit(train_data, train_labels)
    print(model.score(test_data, test_labels))
    model2 = SVC()
    model2.fit(train_data, train_labels)
    print(model2.score(test_data, test_labels))
    model3 = tree.DecisionTreeClassifier()
    model3.fit(train_data, train_labels)
    print(model3.score(test_data, test_labels))

classify_mnist()
