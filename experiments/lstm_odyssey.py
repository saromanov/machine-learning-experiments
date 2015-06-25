import theanets
import theano.tensor as T

#Based on the paper LSTM: A Search Space Odyssey

class LSTMNIG(theanets.layers.Layer):
    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, 4 * self.size)
        self.add_weights('hh', self.size, 4 * self.size)
        self.add_bias('b', 4 * self.size, mean=2)
        #self.add_bias('ci', self.size)
        self.add_bias('cf', self.size)
        self.add_bias('co', self.size)

    def transform(self, inputs):
        def split(z):
            n = self.size
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n], z[:, 3*n:4*n]

        def fn(x_t, h_tm1, c_tm1):
            xi, xf, xc, xo = split(x_t + TT.dot(h_tm1, self.find('hh')))
            #i_t = TT.nnet.sigmoid(xi + c_tm1 * self.find('ci'))
            f_t = TT.nnet.sigmoid(xf + c_tm1 * self.find('cf'))
            c_t = f_t * c_tm1 + T.tanh(xc)
            o_t = TT.nnet.sigmoid(xo + c_t * self.find('co'))
            h_t = o_t * TT.tanh(c_t)
            return [h_t, c_t]

        x = self._only_input(inputs)
        batch_size = x.shape[1]
        (out, cell), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('b')],
            [('h', batch_size), ('c', batch_size)])
        return dict(out=out, cell=cell), updates

class LSTM(theanets.layers.Layer):
    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, 4 * self.size)
        self.add_weights('hh', self.size, 4 * self.size)
        self.add_bias('b', 4 * self.size, mean=2)
        # the three "peephole" weight matrices are always diagonal.
        self.add_bias('ci', self.size)
        self.add_bias('cf', self.size)
        #self.add_bias('co', self.size)

    def transform(self, inputs):
        def split(z):
            n = self.size
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n], z[:, 3*n:4*n]

        def fn(x_t, h_tm1, c_tm1):
            xi, xf, xc, xo = split(x_t + TT.dot(h_tm1, self.find('hh')))
            i_t = TT.nnet.sigmoid(xi + c_tm1 * self.find('ci'))
            f_t = TT.nnet.sigmoid(xf + c_tm1 * self.find('cf'))
            c_t = f_t * c_tm1 + i_t * TT.tanh(xc)
            #o_t = TT.nnet.sigmoid(xo + c_t * self.find('co'))
            h_t = TT.tanh(c_t)
            return [h_t, c_t]

        x = self._only_input(inputs)
        batch_size = x.shape[1]
        (out, cell), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('b')],
            [('h', batch_size), ('c', batch_size)])
        return dict(out=out, cell=cell), updates
 
class GRUMUT(Recurrent):
    ''' Reference to the paper
        An Empirical Exploration of Recurrent Neural Architectures
    '''
    def setup(self):
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xr', self.input_size, self.size)
        self.add_weights('xz', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_weights('hr', self.size, self.size)
        self.add_bias('bh', self.size)
        self.add_bias('br', self.size)
        self.add_bias('bz', self.size)

    def transform(self, inputs):
        def fn(x_t, r_t, z_t, h_tm1):
            z = TT.nnet.sigmoid(x_t)
            r = TT.nnet.sigmoid(r_t + TT.dot(h_tm1, self.find('hr')))
            h_t = TT.tanh(TT.dot(r * h_tm1, self.find('hh')) + x_t)
            return [pre, h_t, z, (1 - z) * h_tm1 + z * h_t]

        x = self._only_input(inputs)
        (pre, hid, rate, out), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('bh'),
             TT.dot(x, self.find('xr')) + self.find('br'),
             TT.dot(x, self.find('xz')) + self.find('bz')],
            [None, None, None, x])
        return dict(pre=pre, hid=hid, rate=rate, out=out), updates

def layer_lstm(n):
    return dict(form='bidirectional', worker='lstm', size=n)

def layer_lstmnig(n):
    return dict(form='bidirectional', worker='lstmnig', size=n)

def layer_grumut(n):
    return dict(form='bidirectional', worker='grumut', size=n)

e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(39, layer_lstmnig(156), layer_lstmnig(300), layer_lstm(102), (51, 'softmax')),
    weighted=True,
)
