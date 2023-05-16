#!/usr/local/bin/python3
import string
import itertools
import numpy as np
import scipy as sp

#from model_util import States,Event
#from Model import *
import Model
#import Utilities
#from Utilities import sort_binary_vectors
from Utilities import sort_binary_vectors,States,Event

#from model_util import states2df,events2df
#from geosse_model_util import *

class GeosseModel(Model.BaseModel):
    
    # initialize model
    def __init__(self, args):
        super().__init__(args)
        self.set_args(args)
        self.set_model()
        return
    
    # assign initial arguments
    def set_args(self, args):
        super().set_args(args)
        self.num_locations = args['num_locations']
        self.num_ranges = 2**self.num_locations - 1
        return
    
    # make model states
    def make_states(self):
        num_locations = self.num_locations
        vec        = [ x for x in itertools.product(range(2), repeat=num_locations) ][1:]
        vec        = sort_binary_vectors(vec)
        letters    = string.ascii_uppercase[0:num_locations]
        lbl        = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
        lbl2vec    = { k:v for k,v in list(zip(lbl,vec)) }
        states     = States(lbl2vec)
        return states

    # make starting state for simulation
    def make_start_state(self):
        # { 'S' : 0 }
        num_ranges = self.num_ranges
        idx = sp.stats.randint.rvs(low=0, high=num_ranges, size=1, random_state=self.rng)[0]
        s = { 'S' : idx }
        return s
    
    # make starting sizes for compartments
    def make_start_sizes(self):
        return {}

    # get all model rates
    def make_params(self, model_variant):
        rates = {}
        
        # get settings
        num_locations = self.num_locations
        rv_fn = self.rv_fn
        rv_arg = self.rv_arg

        # build rates
        if model_variant == 'free_rates':
            params = {
                    'w': rv_fn['w'](size=num_locations, random_state=self.rng, **rv_arg['w']),
                    'e': rv_fn['e'](size=num_locations, random_state=self.rng, **rv_arg['e']),
                    'd': rv_fn['d'](size=num_locations**2, random_state=self.rng, **rv_arg['d']).reshape((num_locations,num_locations)),
                    'b': rv_fn['b'](size=num_locations**2, random_state=self.rng, **rv_arg['b']).reshape((num_locations,num_locations))
                }
            params['x'] = params['x']

        elif model_variant == 'equal_rates':
            params = {
                    'w': np.full(num_locations, rv_fn['w'](size=1, random_state=self.rng, **rv_arg['w'])[0]),
                    'e': np.full(num_locations, rv_fn['e'](size=1, random_state=self.rng, **rv_arg['e'])[0]),
                    'd': np.full((num_locations,num_locations), rv_fn['d'](size=1, random_state=self.rng, **rv_arg['d'])[0]),
                    'b': np.full((num_locations,num_locations), rv_fn['b'](size=1, random_state=self.rng, **rv_arg['b'])[0])
                }
            params['x'] = params['e']

        elif model_variant == 'fig_rates':
            params = {}
        
        return params
    
    # GeoSSE extinction rate
    def make_events_x(self, states, rates):  
        group = 'Extinction'
        events = []
        for i,x in enumerate(states.int2vec):
            if sum(x) == 1:
                for j,y in enumerate(x):
                    if y == 1:
                        name = 'r_x_{i}'.format(i=i)
                        idx = {'i':i}
                        rate = rates[j]
                        ix = [ 'S[{i}]:1'.format(i=i) ]
                        jx = [ 'X' ]
                        e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                        events.append(e)
        return events

    # GeoSSE extirpation rate
    def make_events_e(self, states, rates):
        group = 'Extirpation'
        events = []
        # for each state
        for i,x in enumerate(states.int2vec):
            # for each bit in the state-vector
            for j,y in enumerate(x):
                # if range is size 2+ and region occupied
                if sum(x) > 1 and y == 1:
                    new_bits = list(x).copy()
                    new_bits[j] = 0
                    new_bits = tuple(new_bits)
                    new_state = states.vec2int[ new_bits ]
                    name = 'r_e_{i}_{j}'.format(i=i, j=new_state)
                    rate = rates[j]
                    #xml_str = 'S[{i}]:1 -> S[{j}]:1'.format(i=i, j=new_state)
                    ix = [ 'S[{i}]:1'.format(i=i) ]
                    jx = [ 'S[{j}]:1'.format(j=new_state) ]
                    idx = {'i':i, 'j':new_state}
                    e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                    events.append(e)
        return events

    # GeoSSE within-region speciation rate
    def make_events_w(self, states, rates):
        group = 'Within-region speciation'
        events = []
        # for each state
        for i,x in enumerate(states.int2vec):
            # for each bit in the state-vector
            for j,y in enumerate(x):
                # if region is occupied
                if y == 1:
                    #new_state = j
                    name = 'r_w_{i}_{i}_{j}'.format(i=i, j=j)
                    rate = rates[j]
                    #xml_str = 'S[{i}]:1 -> S[{i}]:1 + S[{j}]:1'.format(i=i, j=new_state)
                    ix = [ 'S[{i}]:1'.format(i=i, j=j) ]
                    jx = [ 'S[{i}]:1'.format(i=i), 'S[{j}]:1'.format(j=j) ]
                    idx = {'i':i, 'j':i, 'k':j}
                    e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                    events.append(e)
        return events

    # GeoSSE dispersal rate
    def make_events_d(self, states, rates):
        group = 'Dispersal'
        # how many compound states
        num_states = len(states.int2int)
        # generate on/off bits
        on  = []
        off = []
        for i,x in enumerate(states.int2vec):
            #print(i,x)
            on.append( [] )
            off.append( [] )
            for j,y in enumerate(x):
                if y == 0:
                    off[i].append(j)
                else:
                    on[i].append(j)
        # convert to dispersal events
        events = []
        for i in range(num_states):
            if sum(on[i]) == 0:
                next
            #print(on[i], off[i])
            for a,y in enumerate(off[i]):
                rate = 0.0
                for b,z in enumerate(on[i]):
                    rate += rates[z][y]
                new_bits = list(states.int2vec[i]).copy()
                new_bits[y] = 1
                #print(states.int2vec[i], '->', new_bits)
                new_state = states.vec2int[ tuple(new_bits) ]
                name = 'r_d_{i}_{j}'.format(i=i, j=new_state)
                ix = [ 'S[{i}]:1'.format(i=i) ]
                jx = [ 'S[{j}]:1'.format(j=new_state) ]
                #xml_str = 'S[{i}]:1 -> S[{j}]:1'.format(i=i, j=new_state)
                #idx = [i, new_state]
                idx = {'i':i, 'j':new_state}
                e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)
        return events

    # GeoSSE between-region speciation rate
    def make_events_b(self, states, rates):
        group = 'Between-region speciation'

        # geometric mean
        def gm(x):
            # unstable: np.power( np.prod(x), 1.0/len(x) )
            return np.exp( np.sum(np.log(x)) / len(x) )
        
        # powerset (to get splits)
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
        
        # compute split rates
        def make_split_rate(left,right,rates):
            r = 0.0
            for i in left:
                for j in right:
                    r += 1.0 / rates[i][j]
            return 1.0/r
            
        # get unique nonempty left/right splits for each range
        events = []
        all_rates = []
        for i,x in enumerate(states.int2set):
            splits = list(powerset(x))[1:-1]
            #print(x)
            for y in splits:
                z = tuple(set(x).difference(y))
                # create event
                left_state = states.set2int[y] 
                right_state = states.set2int[z]
                name = 'r_b_{i}_{j}_{k}'.format(i=i, j=left_state, k=right_state)
                rate = make_split_rate(y,z,rates)
                all_rates.append(rate)
                idx = {'i':i, 'j':left_state, 'k':right_state }
                ix = [ 'S[{i}]:1'.format(i=i) ]
                jx = [ 'S[{j}]:1'.format(j=left_state), 'S[{k}]:1'.format(k=right_state) ]
                #xml_str = 'S[{i}]:1 -> S[{j}]:1 + S[{k}]:1'.format(i=i, j=left_state, k=right_state)
                #e = Event( idx=idx, r=r, n='r_b_{i}_{j}_{k}'.format(i=i, j=left_state, k=right_state), g='Between-region speciation', x=xml_str )
                e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)

        # normalize rates
        #z = gm(all_rates)
        #for i,x in enumerate(events):
        #    events[i].rate = events[i].rate / z

        return events

    # make list of all events in model
    def make_events(self, states, rates):
        events_x = self.make_events_x( states, rates['x'] )
        events_e = self.make_events_e( states, rates['e'] )
        events_d = self.make_events_d( states, rates['d'] )
        events_w = self.make_events_w( states, rates['w'] )
        events_b = self.make_events_b( states, rates['b'] )
        events = events_x + events_e + events_d + events_w + events_b
        return events

    