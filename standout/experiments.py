#!/usr/bin/env python

import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import layers
import holonets
import theano
import theano.tensor as T
import urllib2
import imp
from collections import OrderedDict

def figure3architecture(alpha, beta, batch_size=128, input_dim=784, 
        output_dim=784, n_hidden=2048):
    """
    Returns the final and hidden layers in a network architecture that can be 
    used to replicate the results of Figure 3 in the paper.
    Inputs:
        * alpha - scale hyperparam
        * beta - shift hyperparam
        * batch_size - size of the minibatch used (default is a nice power of 2
    for superstitious reasons)
        * input_dim - dimensionality of input (default is MNIST)
        * output_dim - dimensionaility of output (default is MNIST)
        * n_hidden - number of hidden units to use all the way through
    Outputs:
        * l_hidden, l_out - layers in the network
    """
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop = layers.DropoutAlgorithm2(l_hidden, alpha, beta)
    l_out = lasagne.layers.DenseLayer(l_drop, num_units=output_dim, 
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_hidden, l_out

def make_experiment(l_out, dataset, batch_size=128, N_train=50000, 
        N_valid=10000, N_test=10000):
    """
    Spec up a simple experiment over the MNIST dataset. Autoencoding digits.
    """
    for dset in ['train', 'valid', 'test']:
        dataset['y_'+dset] = dataset['X_'+dset]
    expressions = holonets.monitor.Expressions(l_out, dataset, 
            batch_size=batch_size, update_rule=lasagne.updates.adam, 
            y_tensor_type=T.matrix,
            loss_function=lasagne.objectives.binary_crossentropy,
            loss_aggregate=T.mean, learning_rate=0.001, momentum=0.1)
    # only add channels for loss and accuracy
    for deterministic,dataset in zip([False, True, True],
                                     ["train", "valid", "test"]):
        expressions.add_channel(**expressions.loss(dataset, deterministic))
    channels = expressions.build_channels()
    train = holonets.train.Train(channels, 
            n_batches={'train': N_train//batch_size, 
                       'valid':N_valid//batch_size, 
                       'test':N_test//batch_size})
    loop = holonets.run.EpochLoop(train, dimensions=train.dimensions)
    return loop

# Utility functions from:
# https://github.com/gngdb/variational-dropout/blob/master/varout/experiments.py
def earlystopping(loop, delta=0.01, max_N=1000, verbose=False):
    """
    Stops the expriment once the loss stops improving by delta per epoch.
    With a max_N of epochs to avoid infinite experiments.
    """
    prev_loss, loss_diff = 100, 0.9
    N = 0
    while abs(loss_diff) > delta and N < max_N:
        # run one epoch
        results = loop.run(1)
        N += 1
        current_loss = loop.results["valid Loss"][-1][1]
        loss_diff = (prev_loss-current_loss)/prev_loss
        if verbose:
            print N, loss_diff
        prev_loss = current_loss
    return results

def load_data():
    """
    Standardising data loading; all using MNIST in the usual way:
        * train: 50000
        * valid: 10000
        * test: separate 10000
    """
    # is this the laziest way to load mnist?
    mnist = imp.new_module('mnist')
    exec urllib2.urlopen("https://raw.githubusercontent.com/Lasagne/Lasagne"
            "/master/examples/mnist.py").read() in mnist.__dict__
    dataset = mnist.load_dataset()
    return dict(X_train=dataset[0].reshape(-1, 784),
                y_train=dataset[1],
                X_valid=dataset[2].reshape(-1, 784),
                y_valid=dataset[3],
                X_test=dataset[4].reshape(-1, 784),
                y_test=dataset[5])

