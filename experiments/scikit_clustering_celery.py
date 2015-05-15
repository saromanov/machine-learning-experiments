from celery import Celery
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift

app = Celery('tasks', backend='redis', broker='redis://localhost:6379')

@app.task
def mean_shift(pred):
    meanshift = MeanShift()
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    meanshift = MeanShift()
    meanshift.fit(X)
    result = meanshift.predict(pred)[0]
    return {'result': str(result)}
