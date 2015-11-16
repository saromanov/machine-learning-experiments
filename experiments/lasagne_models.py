import numpy as np
import theano
import theano.tensor as T
import lasagne

class TripletLayer(lasagne.layers.Layer):
    '''
       Networks is simple one layer feed forfard network
    '''
    def __init__(self, inp, num_units, W1=lasagne.init.Normal(.01), 
        W2=lasagne.init.Normal(.01),
        W3 = lasagne.init.Normal(.01),
        **kwargs):
        super(TripletLayer, self).__init__(inp, **kwargs)
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3

    def _loss(self, X1, X2):
        return T.sqr((X1 - X2)**2)

    def get_output_for(self, inp, inpplus, inpminus, **kwargs):
        net1 = T.dot(inp, self.W1)
        net2 = T.dot(inpplus, self.W2)
        net3 = T.dot(inpminus, self.W3)
        first = self._loss(net1, net2)
        second = self._loss(net1, net3)
        dplus = first/(first + second)
        dminus = second/(second + first)
        return dplus + dminus

def l_mlp(epochs):
    inpd = T.tensor4('inputs')
    target = T.ivector('targets')
    inp = lasagne.layers.InputLayer(shape=(None,1,28,28,28))
    inp = lasagne.layers.DimshuffleLayer(inp, (0, 'x', 1,2))
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
    print("Start training..")
    for epoch in range(epochs):
        training_err = 0
        training_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            train_err += train_err(inputs, targets)
            training_batches += 1

l_mlp(20)


