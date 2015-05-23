#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_cifar, plot_layers, plot_images


train, valid, _ = load_cifar()

e = theanets.Experiment(
    theanets.Autoencoder,
    layers=(3072, 1024, 256, 128, 256, 1024, 3072),
)
e.train(train, valid,
        algorithm='layerwise',
        patience=1,
        min_improvement=0.05,
        train_batches=100,
        optimize='adadelta')
e.train(train, valid, min_improvment=0.01, train_batches=200)

plot_layers([e.network.find(i, 'w') for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()

valid = valid[:16*16]
plot_images(valid, 121, 'Sample data')
plot_images(e.network.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()
