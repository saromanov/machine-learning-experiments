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
    loss = loss.mean()
    test_predict = lasagne.layers.get_output(out)
    test_loss = lasagne.objectives.categorical_crossentropy(predict, target)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_predict, axis=1), target), dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([inpd, target], loss, updates=updates)
    val_fn = theano.function([inpd, target], [test_loss, test_acc])
