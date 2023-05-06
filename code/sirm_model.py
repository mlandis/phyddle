#!/usr/local/bin/python3
#from model import *
#from model_util import *
import model_util
#from sirm_model_util import *
#import string
import itertools
import scipy as sp
import numpy as np

from model_util import States,Event
from model_util import states2df,events2df

class SirmModel:
    
    # set up model
    def __init__(self, num_locations, model_variant='equal_rates', feature_set=None):
        
        # create state space
        self.model_type    = 'SIRM'
        self.model_variant = model_variant
        self.num_locations = num_locations
     
        # simulation settings
        self.settings = self.make_settings( self.num_locations )

        # state space
        self.states = self.make_states(self.num_locations)

        # rate space
        self.rates = self.make_rates( self.model_variant, self.settings )

        # event space
        self.events = self.make_events( self.states, self.rates )

        # event space dataframe
        self.df_events = events2df( self.events )

        # state space dataframe
        self.df_states = states2df( self.states )

        # model
        #self.xmlgen = MasterXmlGenerator(self.df_events, self.df_states, self.settings)

    # SIRM simulation settings
    def make_settings(self, num_locations):
        settings = {}

        # generate random starting sizes
        # X ~ Gamma(shape=0.5, scale=1e6) 
        start_sizes = sp.stats.gamma.rvs(size=num_locations, a=0.5, scale=1000000)
        p_start_sizes = start_sizes / np.sum(start_sizes)
        
        # default settings
        settings['num_locations']     = num_locations
        settings['start_state']       = { 'I' : np.random.choice(a=num_locations, size=1, p=p_start_sizes)[0] }
        settings['start_sizes']       = { 'S' : [ int(np.ceil(x)) for x in start_sizes ] }
        settings['sample_population'] = [ 'A' ]
        settings['stop_floor_sizes']  = 0
        settings['stop_ceil_sizes']   = 5000
        settings['rv_fn']             = { 'i':  sp.stats.expon.rvs,
                                          'm': sp.stats.expon.rvs,
                                          'r': sp.stats.expon.rvs,
                                          's':  sp.stats.expon.rvs }
        settings['rv_arg']            = { 'i':  { 'scale' : 0.001 },
                                          'm': { 'scale' : 1.000 },
                                          'r': { 'scale' : 0.010 },
                                          's':  { 'scale' : 0.100 } }
        return settings

    # SIRM state space
    def make_states(self, num_locations):
        # S: Susceptible, I: Infected, R: Recovered, A: Acquired ('Sampled')
        compartments = 'SIRA'
        # Location 0, Location 1, Location 2, ...
        locations = [ str(x) for x in list(range(num_locations)) ]
        # S0, S1, S2, ..., I0, I1, I2, ..., R0, R1, R2, ...
        lbl = [ ''.join(x) for x in list(itertools.product(compartments,locations)) ]
        # 1000, 0100, 0010, 0001 (one-hot encoding for N locations)
        vec = np.identity(num_locations, dtype='int').tolist()
        # { S0:1000, S1:0100, S2:0010, ..., I0:1000, I1:0100, I2:0010, ... }
        lbl2vec = {}
        for i,v in enumerate(lbl):
            j = int(v[1:]) # get location as integer, drop compartment name
            lbl2vec[v] = vec[j]
        # state space object
        states = States(lbl2vec)
        return states

    def make_events(self, states, rates):
        events_i = self.make_events_i( states, rates['i'] )
        events_r = self.make_events_r( states, rates['r'] )
        events_s = self.make_events_s( states, rates['s'] )
        events_m = self.make_events_m( states, rates['m'] )
        events = events_i + events_m + events_r + events_s
        return events

    # SIRM [i]nfection within location
    def make_events_i(self, states, rates):
        group = 'Infect'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] + S[{i}] -> 2 I[{i}]'
            i = int(x[1:])
            name = 'r_I_{i}'.format(i=i)
            idx = {'i':i}
            rate = rates[i]
            ix = [ 'I[{i}]:1'.format(i=i), 'S[{i}]:1'.format(i=i) ]
            jx = [ '2I[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n=, g='Infect', ix=ix, jx=jx )
            events.append(e)

        return events

    # SIRM [r]ecovery within location
    def make_events_r(self, states, rates):
        group = 'Recover'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] -> R[{i}]'
            i = int(x[1:])
            name = 'r_R_{i}'.format(i=i)
            rate = rates[i]
            idx = {'i':i}
            ix = [ 'I[{i}]:1'.format(i=i) ]
            jx = [ 'R[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n=name, g='Recover', ix=ix, jx=jx )
            events.append(e)
            
        return events

    # SIRM [s]ampled from infected host
    def make_events_s(self, states, rates):
        group = 'Sample'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] -> A[{i}]'
            i = int(x[1:])
            name = 'r_S_{i}'.format(i=i)
            idx = {'i':i}
            rate = rates[i]
            ix = [ 'I[{i}]:1'.format(i=i) ]
            jx = [ 'A[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n='r_S_{i}'.format(i=i), g='Sample', ix=ix, jx=jx )
            events.append(e)

        return events

    # SIRM [m]igrates into new population
    def make_events_m(self, states, rates):
        group = 'Migrate'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        state_pairs = list(itertools.product(states_I, states_I))
        for x,y in state_pairs:
            if x != y:
                # 'I[{i}] -> I[{j}]'
                i = int(x[1:])
                j = int(y[1:])
                name = 'r_M_{i}_{j}'.format(i=i, j=j)
                idx = {'i':i, 'j':j}
                rate = rates[i][j]
                ix = [ 'I[{i}]:1'.format(i=i) ]
                jx = [ 'I[{j}]:1'.format(j=j) ]
                e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                # e = Event( idx=idx, r=r, n='r_M_{i}_{j}'.format(i=i, j=j), g='Migrate', ix=ix, jx=jx )
                events.append(e)
        return events

    def make_rates(self, model_variant, settings):
        rates = {}
        
        # get sim RV functions and arguments
        num_locations = settings['num_locations']
        rv_fn = settings['rv_fn']
        rv_arg = settings['rv_arg']

        # all rates within an event type are equal, but rate classes are drawn iid
        if model_variant == 'free_rates':
            # check to make sure arguments in settings are applied to model variant
            rates = {
                'i': rv_fn['i'](size=num_locations, **rv_arg['i']),
                'r': rv_fn['r'](size=num_locations, **rv_arg['r']),
                's': rv_fn['s'](size=num_locations, **rv_arg['s']),
                'm': rv_fn['m'](size=num_locations**2, **rv_arg['m']).reshape((num_locations,num_locations))
            }
        # all rates are drawn iid
        elif model_variant == 'equal_rates':
            rates = {
                'i': np.full(num_locations, rv_fn['i'](size=1, **rv_arg['i'])[0]),
                'r': np.full(num_locations, rv_fn['r'](size=1, **rv_arg['r'])[0]),
                's': np.full(num_locations, rv_fn['s'](size=1, **rv_arg['s'])[0]),
                'm': np.full((num_locations,num_locations), rv_fn['m'](size=1, **rv_arg['m'])[0])
            }
        # e.g. all rates drawn as log-linear functions of features
        elif model_variant == 'feature_rates':
            # other parameters checked for effect-strength of feature on rates
            rates = {
                #'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
                #'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
                #'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
                #'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
            }
        # return rates
        return rates