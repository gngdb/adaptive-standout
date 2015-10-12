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
        * rescale - after dropping out units, rescale other inputs
    """
    def __init__(self, incoming, incoming_beliefnet, rescale=True, **kwargs):
        lasagne.layers.Layer.__init__(self, incoming, **kwargs)
        self.incoming_beliefnet = incoming_beliefnet
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False):
        # get the probabilities from the beliefnet
        self.p = self.incoming_beliefnet.get_output_for(input)
        # sample a Bernoulli matrix using these probabilities
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1. - self.p
            if self.rescale:
                input /= retain_prob
            # sample uniform and threshold to sample from many different 
            # probabilities
            self.uniform = _srng.uniform(self.input_shape)
            # keep if less than retain_prob
            self.mask = self.uniform > retain_prob
            # then just apply mask to input
            return input*self.mask
