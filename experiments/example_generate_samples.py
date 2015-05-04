from sklearn.datasets import load_digits
from generate_samples import GenerateSamples


def example_with_gmm():
    samples = GenerateSamples(load_digits().data)
    samples.kernel_density(50,isplot=True)

def example_with_de():
    samples = GenerateSamples(load_digits().data)
    samples.kernel_density(50,isplot=True)

example_with_gmm()


