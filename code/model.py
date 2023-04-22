#!/usr/local/bin/python3

# libraries
import pandas as pd
import numpy as np
import scipy as sp
#import itertools
#import collections
#import math
#import operator

# model events
class Event:
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, x=None, d=None):
        self.i = None
        self.j = None
        self.k = None
        self.idx = idx
        if 'i' in idx:
            self.i = idx['i']
        if 'j' in idx:
            self.j = idx['j']
        if 'k' in idx:
            self.k = idx['k']
        self.rate = r
        self.name = n
        self.group = g
        self.reaction = x
        self.desc = d
    # make print string
    def make_str(self):
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()



# state space
class StateSpace:
    def __init__(self, lbl2vec):

        # state space dictionary (input)
        self.lbl2vec      = lbl2vec

        # basic info
        self.int2lbl        = list( lbl2vec.keys() )
        self.int2vec        = list( lbl2vec.values() )
        self.int2int        = list( range(len(self.int2vec)) )
        self.int2set        = list( [ tuple([y for y,v in enumerate(x) if v == 1]) for x in self.int2vec ] )
        self.lbl_one        = list( set(''.join(self.int2lbl)) )
        self.num_char       = len( self.int2vec[0] )
        self.num_states     = len( self.lbl_one )

        # relational info
        self.lbl2int = {k:v for k,v in list(zip(self.int2lbl, self.int2int))}
        self.lbl2set = {k:v for k,v in list(zip(self.int2lbl, self.int2set))}
        self.lbl2vec = {k:v for k,v in list(zip(self.int2lbl, self.int2vec))}
        self.vec2int = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2int))}
        self.vec2lbl = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2lbl))}
        self.vec2set = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2set))}
        self.set2vec = {tuple(k):v for k,v in list(zip(self.int2set, self.int2vec))}
        self.set2int = {tuple(k):v for k,v in list(zip(self.int2set, self.int2int))}
        self.set2lbl = {tuple(k):v for k,v in list(zip(self.int2set, self.int2lbl))}
       
    def make_str(self):
        # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
        # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
        s = 'Statespace('
        x = []
        for i in self.int2int:
            # each state in the space is reported as STRING,INT,VECTOR;
            x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
        s += ';'.join(x) + ')'
        return s
    
    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()


class RateSpace:
    def __init__(self, rate_space):
        self.rate_space = rate_space
       
    def make_str(self):
        # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
        # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
        s = 'Ratespace('
        x = []
        #for i in self.int2int:
        # each state in the space is reported as STRING,INT,VECTOR;
        #    x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
        s += ';'.join(x) + ')'
        return s
    
    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()



# event space
class EventSpace:
    def __init__(self, event_space):
        self.event_space = event_space
        

# model itself

class Model:
    # initialization
    def __init__(self, state_space, event_space, rates_to_events, base_rates):
        self.state_space = state_space
        self.event_space = event_space
        self.rates_to_events = rates_to_events
        self.base_rates = base_rates

