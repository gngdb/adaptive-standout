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
        * p - probabilities (output of the belief net)
    """
    def __init__(self, incoming, p, **kwargs):
        lasagne.layers.Layer.__init__(self, incoming, **kwargs)
        # don't like passing both of these, but need to monkey patch the mask
        # later
        self.p = p
        self.mask = sample_mask(self.p)

    def patch_mask(self, mask):
        """
        Patch the mask with whatever you want to put in there. It's like monkey
        patching, but it looks totally legit.
        """
        self.mask = mask 
        return mask

    def get_output_for(self, input, deterministic=False):
        if deterministic or T.mean(self.p) == 0:
            # when making predictions, we take the expectation over the 
            # probabilities
            return self.p*input
        elif self.mask.ndim > 2:
            masked = input.dimshuffle(0,1,'x')*self.mask
            return masked.dimshuffle(2,0,1)
        else:
            return input*self.mask

class DropoutAlgorithm2(lasagne.layers.MergeLayer):
    """
    Algorithm 2 reuses parameters from the previous hidden layer to perform
    the forward prop used for dropout probabilities. Then, these are scaled 
    with alpha and beta. To make this layer work, you must place a 
    DropoutCallForward layer before the layer before this dropout layer. When
    calling lasagne.layers.get_output, the CallForward layer will ensure the
    expression for p is using the correct input variable.
    Inputs:
        * incoming - previous layer
        * alpha - scale hyperparameter
        * beta - shift hyperparameter
    """
    def __init__(self, incoming, alpha, beta, 
            nonlinearity=lasagne.nonlinearities.sigmoid, **kwargs):
        # pull the layer before the incoming layer out of chain 
        incoming_input = lasagne.layers.get_all_layers(incoming)[-2]
        lasagne.layers.MergeLayer.__init__(self, [incoming, incoming_input], **kwargs)
        self.W = incoming.W
        self.alpha = alpha
        self.beta = beta
        self.pi = self.alpha*self.W + self.beta
        self.nonlinearity = nonlinearity

    def get_output_for(self, inputs, deterministic=False):
        self.p = self.nonlinearity(T.dot(inputs[1], self.pi))
        self.mask = sample_mask(self.p)
        if deterministic or T.mean(self.p) == 0:
            return self.p*inputs[0]
        else:
            return inputs[0]*self.mask

    def get_output_shape_for(self, input_shapes):
        """
        Layer will always return the input shape of the incoming layer, because
        it's just applying a mask to that layer.
        """
        return input_shapes[0]

def sample_mask(p):
    """
    give a matrix of probabilities, this will sample a mask. Theano.
    """
    # sample uniform and threshold to sample from many different 
    # probabilities
    uniform = _srng.uniform(p.shape)
    # keep if less than retain_prob
    mask = uniform < p
    return mask

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

