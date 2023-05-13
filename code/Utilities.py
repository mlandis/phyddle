# general libraries
import argparse
import importlib
import pandas as pd
import numpy as np


##################
# Helper Classes #
##################

# model events
class Event:
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
        self.i = -1
        self.j = -1
        self.k = -1
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
        self.ix = ix
        self.jx = jx
        self.reaction = ' + '.join(ix) + ' -> ' + ' + '.join(jx)
        
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
class States:
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
        self.int2vecstr = [ ''.join([str(y) for y in x]) for x in self.int2vec ]
        self.vecstr2int = { v:i for i,v in enumerate(self.int2vecstr) }
       
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
    
    def make_df(self):
        df = pd.DataFrame()



#################
# FILE HANDLERS #
#################

def load_config(config_fn, arg_overwrite=True):
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config')
    parser.add_argument('--cfg', dest='config_fn', type=str, help='Config file name')
    parser.add_argument('--job', dest='job_name', type=str, help='Job directory')
    parser.add_argument('--start_idx', dest='start_idx', type=int, help='Start index for simulation')
    parser.add_argument('--end_idx', dest='end_idx', type=int, help='End index for simulation')
    args = parser.parse_args()
    
    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn != None:
        config_fn = args.config_fn
    
    # config from file
    m = importlib.import_module(config_fn)

    # overwrite default args
    if args.job_name != None:
        m.my_all_args['job_name'] = args.job_name
    if args.start_idx != None:
        m.my_sim_args['start_idx'] = args.start_idx
    if args.end_idx != None:
        m.my_sim_args['end_idx'] = args.end_idx

    # update args    
    m.my_sim_args = m.my_sim_args | m.my_all_args
    m.my_fmt_args = m.my_fmt_args | m.my_all_args
    m.my_lrn_args = m.my_lrn_args | m.my_all_args

    # return new args
    return m



#################
# MODEL HELPERS #
#################

def make_symm(m):
    d = np.diag(m)
    m = np.triu(m)
    m = m + m.T
    np.fill_diagonal(m, d)
    return m

# Chat-GPT function
def sort_binary_vectors(binary_vectors):
    """
    Sorts a list of binary vectors in order of number of "on" bits first, and then left to right in terms of which bits are "on".
    """
    # Define a helper function to count the number of "on" bits in a binary vector
    def count_ones(binary_vector):
        return sum(binary_vector)
    
    # Sort the binary vectors in the list first by number of "on" bits
    sorted_vectors = sorted(binary_vectors, key=count_ones)
    
    # Sort the binary vectors in the list by "on" bits from left to right
    for i in range(len(sorted_vectors)):
        for j in range(i+1, len(sorted_vectors)):
            if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
                for k in range(len(sorted_vectors[i])):
                    if sorted_vectors[i][k] != sorted_vectors[j][k]:
                        if sorted_vectors[j][k] > sorted_vectors[i][k]:
                            sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
                        break
                
    return sorted_vectors

def events2df(events):
    df = pd.DataFrame({
        'name'     : [ e.name for e in events ],
        'group'    : [ e.group for e in events ], 
        'i'        : [ e.i for e in events ],
        'j'        : [ e.j for e in events ],
        'k'        : [ e.k for e in events ],
        'reaction' : [ e.reaction for e in events ],
        'rate'     : [ e.rate for e in events ]
    })
    return df

def states2df(states):
    df = pd.DataFrame({
        'lbl' : states.int2lbl,
        'int' : states.int2int,
        'set' : states.int2set,
        'vec' : states.int2vec
    })
    return df
