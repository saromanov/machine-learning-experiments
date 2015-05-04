from sklearn.neighbors.kde import KernelDensity
import numpy as np
from sklearn.datasets import load_digits
from sklearn import mixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from skimage import io
#from skimage.color import rgb2gray
#from skimage.transform import resize
import pylab as pl
import math
import logging
import os


#Generate additional samples from datasets

class NADE:
    def __init__(self, x, nvis, nhid, batch_size):
        nvis = x.shape[0]
        self.x = x
        self.nvis = nvis
        self.nhid = nhid
        self.W = (np.random.random((nhid, nvis)) - 0.5) * 0.01
        self.V = (np.random.random((nhid, nvis)) - 0.5) * 0.01
        self.b = np.ones((nhid, ))
        self.b2 = np.ones((nvis, ))

    def one_step(self, a, W_i, V_i, x_i, b, prob):
        h_i = self._sigm(a)
        pg = self._sigm(np.dot(h_i, W_i))
        prob = prob + np.log(pg * x_i + (1 - pg) * (1 - x_i))
        a += np.outer(x_i, V_i)
        #print(result.shape, V_i.shape, x_i.shape)
        return a, prob

    def _sigm(self, x):
        return 1/(1 + np.exp(-x))

    def sample(self, num):
       """ Sample <num> first elements from distribution
       """
       result = self.trainNP()
       return result
    def trainNP(self):
        #prob = np.zeros((self.nvis,))
        prob = np.zeros(self.nvis)
        a = np.zeros((self.nvis, self.nhid))
        for i in range(1, self.nhid):
            a, prob = self.one_step(a, self.W.T[i], self.V[i], self.x.T[i], self.b, prob)
        return prob

class GenerateSamples:
    def __init__(self, dataset):
	    self.shape = dataset[0].shape
	    self.dataset = dataset
	    if len(self.shape) == 2:
		    self.flatshape = self.shape[0] * self.shape[1]
		    self.dataset = np.reshape(dataset, (self.flatshape, ))

    def _prepare_data(self, data):
        pca = PCA(n_components=15, whiten=False)
        return pca, pca.fit_transform(data)

    def kernel_density(self, num, isplot=True):
        num-=1
        """ Generate samples with kernel density estimation """
	return self._compute(KernelDensity(kernel='gaussian', bandwidth=0.2), num, isplot=isplot)

    def gmm(self, num, isplot=True):
        num_components = 100
        num-=1
        if num < 100:
            num_components = num
        return self._compute(mixture.GMM(num_components, covariance_type='full'), num, isplot=isplot)

    def _compute(self, g, num, isplot=True):
        pca, new_data = self._prepare_data(self.dataset)
	g.fit(new_data)
	samples = g.sample(num)
	samples = pca.inverse_transform(samples)
	if isplot:
            firstnum = int(abs(math.sqrt(num)))
            power = firstnum**2
            secondnum = (num - power) + firstnum
            fig, ax = plt.subplots(firstnum, secondnum, subplot_kw=dict(xticks=[], yticks=[]))
            counter = 0
            for i in range(firstnum):
	        for j in range(secondnum):
		    im = ax[i, j].imshow(np.reshape(samples[counter], (8,8)),cmap=plt.cm.binary, interpolation='nearest')
		    counter += 1
            plt.show()
        return samples




def test_NADE():
	learning_rate = 0.01
	training_epoch = 20
	#dataset = 'mnist.pk1.gz'
	#datasets = load_data(dataset)
	dataset = load_digits()
	#train_set_X, train_set_Y = datasets[0]
	rng = np.random.RandomState(123)
	n = NADE(dataset.data[0:100],64,20,0.1)
	img = n.sample(10)
	print(img)
	#pl.imshow(np.reshape(img, (8,8)), cmap=pl.cm.gray_r, interpolation='nearest')
	#pl.show()



def load_birds(path):
    if not os.path.exists(path):
        msg = "Path not found"
        logging.fatal(msg)
        raise Exception(msg)
    for value in os.listdir(path):
        fullpath = path + '/' + value
        result = np.array(rgb2gray(resize(io.imread(fullpath), (8,8))))
        yield np.reshape(result, (8*8,))




