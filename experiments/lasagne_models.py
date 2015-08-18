import numpy as np
import theano
import theano.tensor as T
import lasagne


def l_mlp():
    inpd = T.tensor4('inputs')
    target = T.ivector('targets')
    inp = lasagne.layers.InputLayer(shape=(None,1,28,28,28), input_var=input_var)
    drop = lasagne.layers.GaussianNoiseLayer(inp, sigma=0.15)
    h1 = lasagne.layers.DenseLayer(drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify)
    drop2 = lasagne.layers.DropoutLayer(inp, p=0.5)
    out = lasagne.layers.DenseLayer(drop2, num_units=10, nonlinearities = lasagne.nonlinearities.softmax)

    predict = lasagne.layers.get_output(out)
    loss = lasagne.objectives.categorical_crossentropy(predict, target)