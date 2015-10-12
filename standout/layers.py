#!/usr/bin/env python

import lasagne.layers
import theano
import theano.tensor as T

class Dense(lasagne.layers.DenseLayer):
    """
    Dense layer implementing the binary belief network part of the Standout
    algorithm. This encodes the dropout probabilities as described in equation
    (2) of the paper:

    P(m_j = 1 | {a_i : i<j}) = f(\sum_{i: i<j} \pi_{j,i} a_i)

    Used in conjunction with a dropout layer that can take the expressions 
    representing the dropout probabilities and apply them to each unit.
    """

class Dropout(lasagne.layers.Dropout):
    """
    Standout version of dropout. Operates in the same way as traditional 
    dropout, but this implementation allows a different dropout probability
    for each unit; and takes as an input the expression for each of these
    dropout probability.
    """
