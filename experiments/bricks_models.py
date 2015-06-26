from blocks.bricks import Linear, Softmax, Logistic, MLP
from blocks.algorithms import GradientDescent, Momentum, AdaGrad, Scale
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
import theano.tensor as T
import theano

def m_Linear(name, inp, out):
    return Linear(name=name, input_dim=inp, output_dim=out)

def Custom_MLP():
    mnist = MNIST(("train",))
    x = T.matrix('x')
    y = T.lmatrix('y')
    layer1 = m_Linear("hidden", 784, 200)
    hidden = Logistic.apply(layer1.apply(x))
    layer2 = m_Linear("output", 200, 10)
    output = Softmax.apply(layer2.apply(hidden))
    layer1.weights_init = IsotropicGaussian(.01)
    layer1.biases_init = Constant(0)
    layer2.weights_init = IsotropicGaussian(.01)
    layer2.biases_init = Constant(0)
    layer1.initialize()
    layer2.initialize()
    loss = CategoricalCrossEntropy.apply(y.flatten(), output)
    gr = ComputationGraph(loss)
    data_stream = Flatten(DataStream.default_stream(mnist, 
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    algorithm = GradientDescent(cost=cost, step_rule=Scale(learning_rate=0.1), params=gr.parameters)
    loop = MainLoop(data_stream=data_stream, algorithm=algorithm)
    loop.run()
