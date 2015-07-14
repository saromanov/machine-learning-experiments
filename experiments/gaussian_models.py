from sklearn import mixture
from sklearn.datasets import fetch_mldata

def companies():
    data = fetch_mldata('golfdata')
    gmm = mixture.GMM(n_components=5, covariance_type='full')
    gmm.fit(data.data)

companies()