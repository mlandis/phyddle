#!/usr/local/bin/python3

# libraries
import pandas as pd
import numpy as np
import scipy as sp

# model events
class Event:
    # initialize
    def __init__(self, i=None, j=None, k=None, r=0.0, n=None, t=None, rxn=None, d=None):
        self.i = i
        self.j = j
        self.k = k
        self.rate = r
        self.name = n
        self.type = t
        self.rxn = rxn
        self.desc = d
    # make print string
    def make_str(self):
        s = 'Event({name},{type},{rate},{i}'.format(name=self.name, type=self.type, rate=self.rate, i=self.i)        
        if self.j is not None:
            s += ',{j}'.format(j=self.j)
        if self.k is not None:
            s += ',{k}'.format(k=self.k)
        s += ')'
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
        self.lbl_one        = list( set(''.join(self.int2lbl)) )
        self.num_char       = len( self.int2vec[0] )
        self.num_states     = len( self.lbl_one )

        # relational info
        self.int2lbl = {k:v for k,v in list(zip(self.int2int, self.int2lbl))}
        self.lbl2int = {k:v for k,v in list(zip(self.int2lbl, self.int2int))}
        self.vec2int = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2int))}
        self.vec2lbl = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2lbl))}
       
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


# single event
evt = []
evt.append( Event(0, None, None, 0.3, 'r_e_0', 'extinction') )
evt.append( Event(0, 1, None, 0.4, 'r_d_0_1', 'dispersal') )
evt.append( Event(2, 0, 1, 0.2, 'r_b_2_0_1', 'extinction') )
print(evt)

# state space
vec       = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1,], [1,1,1] ]
lbl       = [ 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC' ]
lbl2vec = { k:v for k,v in list(zip(lbl,vec)) }
ss = StateSpace(lbl2vec)
print(ss)

# rate space
num_char = 3
rates = {
    'r_w': sp.stats.expon.rvs(size=num_char),
    'r_e': sp.stats.expon.rvs(size=num_char), 
    'r_d': sp.stats.expon.rvs(size=num_char**2).reshape((num_char**2,1)),
    'r_b': sp.stats.expon.rvs(size=num_char**2).reshape((num_char**2,1))
}
rates['r_x'] = rates['r_e']
rs = RateSpace(rates)
print(rs)



# event-rate mapping functions
def fn_d(s, r):
    
    # how many compound states
    num_states = len(s.int2int)

    # generate on/off bits
    on  = []
    off = []
    for i,x in enumerate(s.int2vec):
        print(i,x)
        on.append( [] )
        off.append( [] )
        for j,y in enumerate(x):
            #print(i,j,x,y)
            if y == 0:
                off[i].append(j)
            else:
                on[i].append(j)

    # convert to dispersal events
    e = []
    for i in range(num_states):
        for a,y in enumerate(on):
            for b,z in enumerate(off):
                j = i
                r = 0.0

                Event(i, j, None, 0.0, 'r_d_{i}_{j}'.format(i=i,j=j), '' )

    return on,off

fn_d(ss, rates)

# event space
df = pd.DataFrame()
event_classes     = [ 'dispersal', 'extirpation', 'extinction', 'within-region speciation', 'between-region speciation' ]
event_anaclado    = [ 'transition', 'transition', 'death', 'speciation', 'speciation' ]
event_num_index   = [ 2, 1, 1, 1, 3 ]
event_rules    = [ fn_d, fn_e, fn_x, fn_w, fb_b ]