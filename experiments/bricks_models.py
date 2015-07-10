from blocks.bricks import Linear, Softmax, Logistic, MLP, Rectifier
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, MaxPooling, Flattener
from blocks.algorithms import GradientDescent, Momentum, AdaGrad, AdaDelta, Scale
from blocks.bricks.cost import CategoricalCrossEntropy, SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing
from blocks.main_loop import MainLoop
from fuel.datasets import MNIST, CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
import theano.tensor as T
import theano

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
    mnist =CIFAR10(("train", ))
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
    data_stream = Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    algorithm = GradientDescent(cost=loss, step_rule=AdaDelta(), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
    loop.run()
    #func = theano.function([x], res)
    #seq = ConvolutionalSequence([conv1], 3, image_size=(28,28))

Custom_conv()