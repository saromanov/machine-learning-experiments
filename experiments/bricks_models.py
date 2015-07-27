from blocks.bricks import Linear, Softmax, Logistic, MLP, Rectifier, Tanh, LinearMaxout, Initializable,\
Feedforward, Brick
from blocks.bricks import application
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, MaxPooling, Flattener
from blocks.algorithms import GradientDescent, Momentum, AdaGrad, AdaDelta, Scale, Adam
from blocks.bricks.cost import CategoricalCrossEntropy, SquaredError
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.graph import ComputationGraph
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, FinishAfter
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_nans
from blocks.bricks.recurrent import GatedRecurrent, SimpleRecurrent, LSTM
from fuel.datasets import MNIST, CIFAR10, IterableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
import theano.tensor as T
import theano
import numpy as np


class Fun(Initializable):
    def __init__(self, inpnum, outnum, **kwargs):
        super(Fun, self).__init__(self, **kwargs)
        self.inpnum = inpnum
        self.outnum = outnum
        encoder = MLP(activations=[Logistic(), Softmax()], dims=[784,inpnum,outnum])
        decoder = MLP(activations=[Tanh(), Softmax()], dims=[outnum,inpnum,784])
        encoder.weights_init = IsotropicGaussian(.01)
        encoder.biases_init = Constant(0)
        decoder.weights_init = IsotropicGaussian(.01)
        decoder.biases_init = Constant(0)
        self.encoder = encoder
        self.decoder = decoder
        self.loss = CategoricalCrossEntropy()
        self.children = [encoder, decoder, self.loss]

    @application
    def cost(self, x):
        result = self.encoder.apply(x)
        decoder_res = self.decoder.apply(result)
        return self.loss.apply(x.flatten(), decoder_res.flatten())


def m_Linear(name, inp, out):
    return Linear(name=name, input_dim=inp, output_dim=out)

def test_set_monitor():
    mnist = MNIST(("test", ))
    return Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=1024)))

def test_set_monitor_cifar():
    mnist = CIFAR10(("test", ))
    return Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=1024)))

def Custom_MLP():
    mnist = MNIST(("train",))
    x = T.matrix('features')
    y = T.lmatrix('targets')
    layer1 = m_Linear("hidden", 784, 200)
    hidden = Logistic().apply(layer1.apply(x))
    layer2 = m_Linear("output", 200, 10)
    output = Softmax().apply(layer2.apply(hidden))
    layer1.weights_init = IsotropicGaussian(.01)
    layer1.biases_init = Constant(0)
    layer2.weights_init = IsotropicGaussian(.01)
    layer2.biases_init = Constant(0)
    layer1.initialize()
    layer2.initialize()
    loss = CategoricalCrossEntropy().apply(y.flatten(), output)
    gr = ComputationGraph(loss)
    monitor = DataStreamMonitoring(variables=[loss], data_stream=test_set_monitor())
    data_stream = Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    algorithm = GradientDescent(cost=loss, step_rule=Scale(learning_rate=0.1), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
    loop.run()

def Custom_MLP2():
    mnist = CIFAR10("train",)
    x = T.matrix('features')
    y = T.lmatrix('targets')
    layer1 = m_Linear("hidden", 784, 300)
    hidden = Rectifier().apply(layer1.apply(x))
    layer2 = m_Linear("hidden2", 300, 50)
    hidden2 = Rectifier().apply(layer2.apply(hidden))
    layer3 = m_Linear("output", 50, 10)
    output = Softmax().apply(layer3.apply(hidden2))
    layer1.weights_init = IsotropicGaussian(.01)
    layer1.biases_init = Constant(0)
    layer2.weights_init = IsotropicGaussian(.01)
    layer2.biases_init = Constant(0)
    layer3.weights_init = IsotropicGaussian(.01)
    layer3.biases_init = Constant(0)
    layer1.initialize()
    layer2.initialize()
    layer3.initialize()
    loss = CategoricalCrossEntropy().apply(y.flatten(), output)
    gr = ComputationGraph(loss)
    monitor = DataStreamMonitoring(variables=[loss], data_stream=test_set_monitor())
    data_stream = Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    algorithm = GradientDescent(cost=loss, step_rule=AdaDelta(), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
    loop.run()

def Custom_conv():
    cifar10 =MNIST(("train", ))
    x = T.tensor4('features')
    y = T.lmatrix('targets')
    act = Rectifier().apply
    conv1 = ConvolutionalLayer(act, (32,32), 3, (2,2), 4, image_size=(32,32), batch_size=256, name='conv1',
        biases_init=Constant(0), weights_init=Constant(1.), tied_biases=True)
    conv1.initialize()
    res = conv1.apply(x)
    flat = Flattener()
    out = flat.apply(res)
    mlp = MLP(activations=[Logistic(), Softmax()], dims=[2056, 100, 10])
    loss = Softmax().categorical_cross_entropy(y, mlp.apply(out))
    gr = ComputationGraph(loss)
    monitor = DataStreamMonitoring(variables=[loss], data_stream=test_set_monitor_cifar())
    data_stream = Flatten(DataStream.default_stream(cifar10, 
        iteration_scheme=SequentialScheme(cifar10.num_examples, 10)))
    algorithm = GradientDescent(cost=loss, step_rule=AdaDelta(), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
    loop.run()
    #func = theano.function([x], res)
    #seq = ConvolutionalSequence([conv1], 3, image_size=(28,28))

def custum_rnn():
    x = T.tensor3('x')
    y = T.tensor3('y')
    seq1 = np.random.randn(100, 50, 10, 2)
    seq2 = np.zeros((100, 50, 10, 2))
    inp = m_Linear('inp', seq1.shape[-1], 5)
    inp.weights_init = IsotropicGaussian(.01)
    inp.biases_init = Constant(0)
    inpout = inp.apply(x)
    rnn = SimpleRecurrent(activation=Tanh(), dim=5, name="RNN1")
    rnn.weights_init = IsotropicGaussian(.01)
    rnn.biases_init = Constant(0)
    hidden = rnn.apply(inpout)
    output_layer = Linear(name='out', input_dim=5, output_dim=seq2.shape[-1])
    output_layer.weights_init = IsotropicGaussian(.01)
    output_layer.biases_init = Constant(0)
    output = output_layer.apply(hidden)

    cost = CategoricalCrossEntropy().apply(output, y)
    gr = ComputationGraph(cost)
    algo = GradientDescent(cost=cost, step_rule=AdaDelta(), params=gr.parameters)
    inp.initialize()
    rnn.initialize()
    output_layer.initialize()

    dataset = IterableDataset({'x': seq1, 'y': seq2})
    stream = DataStream(dataset)
    loop = MainLoop(data_stream=stream, algorithm=algo, extensions=[Printing(), FinishAfter(after_n_epochs=10)])
    loop.run()

def custum_gru():
    x = T.tensor3('x')
    h0 = T.tensor3('h0')
    g = T.tensor3('g')
    y = T.tensor3('y')
    seq1 = np.random.randn(100, 50, 10, 2)
    seq2 = np.zeros((100, 50, 10, 2))
    inp = m_Linear('inp', seq1.shape[-1], 5)
    inp.weights_init = IsotropicGaussian(.01)
    inp.biases_init = Constant(0)
    inpout = inp.apply(x)
    gru = GatedRecurrent(activation=Tanh(), dim=5, name="GRU1")
    gru.weights_init = IsotropicGaussian(.01)
    gru.biases_init = Constant(0)
    hidden = gru.apply(x, h0,g)
    output_layer = Linear(name='out', input_dim=5, output_dim=seq2.shape[-1])
    output_layer.weights_init = IsotropicGaussian(.01)
    output_layer.biases_init = Constant(0)
    output = output_layer.apply(hidden)
    cost = CategoricalCrossEntropy().apply(output, y)
    gr = ComputationGraph(cost)
    algo = GradientDescent(cost=cost, step_rule=Adam(), params=gr.parameters)
    inp.initialize()
    gru.initialize()
    output_layer.initialize()


def triplet():
    mnist = MNIST(("train",))
    x = T.matrix('features')
    y = T.lmatrix('targets')
    mlp1 = MLP(activations=[Logistic(), Softmax()], dims=[784,300,10])
    mlp1out = mlp1.apply(x)
    mlp1.weights_init = IsotropicGaussian(.01)
    mlp1.biases_init = Constant(0)
    mlp2 = MLP(activations=[Rectifier(), Softmax()], dims=[784,300,1])
    mlp2out = mlp1.apply(mlp1out)


def fun_test():
    mnist = MNIST(("train",))
    x = T.matrix('features')
    fun1 = Fun(300,10)
    loss = fun1.cost(x)
    gr = ComputationGraph(loss)
    monitor = DataStreamMonitoring(variables=[loss], data_stream=test_set_monitor())
    data_stream = Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    algorithm = GradientDescent(cost=loss, step_rule=Scale(learning_rate=0.1), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
    loop.run()

fun_test()