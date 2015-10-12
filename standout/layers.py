#!/usr/bin/env python

import lasagne.layers
import theano
import theano.tensor as T

class Dropout(lasagne.layers.Dropout):
    """
    Standout version of dropout. Operates in the same way as traditional 
    dropout, but this implementation allows a different dropout probability
    for each unit; and takes as an input the expression for each of these
    dropout probability.
    """
