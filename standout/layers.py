#!/usr/bin/env python

import lasagne.layers
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams
_srng = MRG_RandomStreams(42)

class Dropout(lasagne.layers.Layer):
    """
    Standout version of dropout. Operates in the same way as traditional 
    dropout, but this implementation allows a different dropout probability
    for each unit; and takes as an input the expression for each of these
    dropout probabilities.
    Inputs:
        * incoming - previous layer in the network.
        * incoming_beliefnet - layer encoding dropout probabilities.
        * alpha - hyperparameter controlling scaling
        * beta - hyperparameter controlling bias
        * rescale - after dropping out units, rescale other inputs
    """
    def __init__(self, incoming, incoming_beliefnet, alpha=1.0, beta=0.0, 
            rescale=True, **kwargs):
        lasagne.layers.Layer.__init__(self, incoming, **kwargs)
        self.incoming_beliefnet = incoming_beliefnet
        self.rescale = rescale
        self.alpha = alpha
        self.beta = beta

    def get_output_for(self, input, deterministic=False):
        # get the probabilities from the beliefnet
        self.p = self.alpha*self.incoming_beliefnet
               + self.beta
        # sample a Bernoulli matrix using these probabilities
        if deterministic or self.p == 0:
            # when making predictions, we take the expectation over the 
            # probabilities
            return self.p*input
        else:
            retain_prob = 1. - self.p
            # sample uniform and threshold to sample from many different 
            # probabilities
            self.uniform = _srng.uniform(self.input_shape)
            # keep if less than retain_prob
            self.mask = self.uniform > retain_prob
            # then just apply mask to input
            return input*self.mask

def get_all_beliefnets(output_layer, input_var):
    """
    Takes an output layer and collects up all the belief net layers in the 
    network. Useful for gathering up the parameters so that they can then 
    be updated.
    """
    all_layers = lasagne.layers.get_all_layers(output_layer, input_var)
    all_beliefnets = [l.incoming_beliefnet for l in all_layers 
                      if isinstance(l, Dropout)]
    return all_beliefnets

def get_all_parameters(output_layer):
    """
    Analogous to the lasagne.layers.get_all_parameters, but gathers only the 
    parameters of the standout layers.
    """
    all_beliefnets = get_all_beliefnets(output_layer)
    return [l.W for l in all_beliefnets]
