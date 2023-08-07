#!/usr/bin/env python
"""
SirmModel
=========
Defines a class for a Susceptible-Infectious-Recovered + Migration (SIRM) model.
Dervies from phyddle.Model.BaseModel.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import itertools
import scipy as sp
import numpy as np
import pandas as pd

from models.model import BaseModel
from master_util import States,Event


model_type = 'SIRM'
model_variants = {
    'free_rates' : {
        'desc':'all event rates have independent values',
        'params':['R0','S0','sampling','recovery','migration']
    },
    'equal_rates' : {
        'desc':'all event rates of type share one value',
        'params':['R0','S0','sampling','recovery','migration']
    },
    'vistor' : {
        'desc':'visitor model',
        'params':['R0','S0','sampling','recovery','migration','return']
    }
}

variant_registry = []
variant_registry_names = ['variant_name',      'description'  ] 
for k,v in model_variants.items():
    variant_name = k
    variant_desc = v['desc']
    variant_registry.append([variant_name, variant_desc])

variant_registry = pd.DataFrame( variant_registry, columns = variant_registry_names)

class SirmModel(BaseModel):

    # initialize model
    def __init__(self, args):
        super().__init__(args)
        self.set_args(args)
        self.set_model(None)
        return
    
    # assign initial arguments
    def set_args(self, args):
        super().set_args(args)
        self.num_char = args['num_char']
        self.num_states = args['num_states']
        return
    
    def set_model(self, idx):
        super().set_model(idx)
        return

    # SIRM state space
    def make_states(self):
        #num_char = self.num_char
        num_states = self.num_states
        # S: Susceptible, I: Infected, R: Recovered, A: Acquired ('Sampled')
        compartments = 'SIRA'
        # Location 0, Location 1, Location 2, ...
        locations = [ str(x) for x in list(range(num_states)) ]
        # S0, S1, S2, ..., I0, I1, I2, ..., R0, R1, R2, ...
        lbl = [ ''.join(x) for x in list(itertools.product(compartments,locations)) ]
        # 1000, 0100, 0010, 0001 (one-hot encoding for N locations)
        vec = np.identity(num_states, dtype='int').tolist()
        # { S0:1000, S1:0100, S2:0010, ..., I0:1000, I1:0100, I2:0010, ... }
        lbl2vec = {}
        for i,v in enumerate(lbl):
            j = int(v[1:]) # get location as integer, drop compartment name
            lbl2vec[v] = vec[j]
        # state space object
        states = States(lbl2vec)
        return states
        
    def make_start_conditions(self):
        # p_start_sizes = self.start_sizes['S'] / np.sum(self.start_sizes['S'])
        # start_state = list(sp.stats.multinomial.rvs(n=1, p=p_start_sizes, random_state=self.rng)).index(1)
        start_state = {}
        start_sizes = {}
        start_state['I'] = self.params['start_state'][0]
        start_sizes['S'] = self.params['S0']

        return start_state, start_sizes
        
    def make_params(self, model_variant):
        params = {}

        # get sim RV functions and arguments
        num_states = self.num_states
        rv_fn = self.rv_fn
        rv_arg = self.rv_arg

        # all rates within an event type are equal, but rate classes are drawn iid
        if model_variant == 'free_rates':
            # check to make sure arguments in settings are applied to model variant
            params = {
                'S0'        : rv_fn['S0'](size=num_states, random_state=self.rng, **rv_arg['S0']),
                'R0'        : rv_fn['R0'](size=num_states, random_state=self.rng, **rv_arg['R0']),
                'sampling'  : rv_fn['sampling'](size=num_states, random_state=self.rng, **rv_arg['sampling']),
                'recovery'  : rv_fn['recovery'](size=num_states, random_state=self.rng, **rv_arg['recovery']),
                'migration' : rv_fn['migration'](size=num_states**2, random_state=self.rng, **rv_arg['migration']).reshape((num_states,num_states))
            }
            params['S0'] = int( np.round(params['S0']) )
            params['infection'] = params['R0'] / (params['recovery'] + params['sampling']) * (1. / params['S0'])
            p_start_sizes = params['S0'] / np.sum(params['S0'])
            start_state = sp.stats.multinomial.rvs(n=1, p=p_start_sizes, random_state=self.rng)
            params['start_state'] = np.where( start_state==1 )[0]
            #monitors = { 'i':rates['i'], 'r':rates['r'], 's':rates['s'], 'm':rates['m'] }
            
        # all rates are drawn iid
        elif model_variant == 'equal_rates':
            params = {
                'S0'        : np.full(num_states, rv_fn['S0'](size=1, random_state=self.rng, **rv_arg['S0'])[0]),
                'R0'        : np.full(num_states, rv_fn['R0'](size=1, random_state=self.rng, **rv_arg['R0'])[0]),
                'sampling'  : np.full(num_states, rv_fn['sampling'](size=1, random_state=self.rng, **rv_arg['sampling'])[0]),
                'recovery'  : np.full(num_states, rv_fn['recovery'](size=1, random_state=self.rng, **rv_arg['recovery'])[0]),
                'migration' : np.full((num_states,num_states),
                                      rv_fn['migration'](size=1, random_state=self.rng, **rv_arg['migration'])[0])
            }
            params['S0'] = np.round(params['S0'])
            # R0 = infection/(recovery+sampling) so infection = R0*(recovery+sampling)
            params['infection'] = params['R0'] * (params['recovery'] + params['sampling']) * (1. /  params['S0'])
            p_start_sizes = params['S0'] / np.sum(params['S0'])
            start_state = sp.stats.multinomial.rvs(n=1, p=p_start_sizes, random_state=self.rng)
            params['start_state'] = np.where( start_state==1 )[0]
            #monitors = { 'i':rates['i'][0],'r':rates['r'][0],'s':rates['s'][0],'m':rates['m'][0][1] }

        # e.g. all rates drawn as log-linear functions of features
        elif model_variant == 'feature_rates':
            # other parameters checked for effect-strength of feature on rates
            params = {
                #'Infect': np.full(num_states, sp.stats.expon.rvs(size=1)[0]),
                #'Recover': np.full(num_states, sp.stats.expon.rvs(size=1)[0]), 
                #'Sample': np.full(num_states, sp.stats.expon.rvs(size=1)[0]), 
                #'Migrate': np.full((num_states,num_states), sp.stats.expon.rvs(size=1)[0])
            }
        # return rates
        #print(rates)
        return params

    def make_events(self, states, params):
        events_i = self.make_events_i( states, params['infection'] )
        events_r = self.make_events_r( states, params['recovery'] )
        events_s = self.make_events_s( states, params['sampling'] )
        events_m = self.make_events_m( states, params['migration'] )
        events = events_i + events_m + events_r + events_s
        return events

    # SIRM [i]nfection within location
    def make_events_i(self, states, params):
        group = 'Infect'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] + S[{i}] -> 2 I[{i}]'
            i = int(x[1:])
            name = 'r_I_{i}'.format(i=i)
            idx = {'i':i}
            rate = params[i]
            ix = [ 'I[{i}]:1'.format(i=i), 'S[{i}]:1'.format(i=i) ]
            jx = [ '2I[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n=, g='Infect', ix=ix, jx=jx )
            events.append(e)

        return events

    # SIRM [r]ecovery within location
    def make_events_r(self, states, params):
        group = 'Recover'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] -> R[{i}]'
            i = int(x[1:])
            name = 'r_R_{i}'.format(i=i)
            rate = params[i]
            idx = {'i':i}
            ix = [ 'I[{i}]:1'.format(i=i) ]
            jx = [ 'R[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n=name, g='Recover', ix=ix, jx=jx )
            events.append(e)
            
        return events

    # SIRM [s]ampled from infected host
    def make_events_s(self, states, params):
        group = 'Sample'
        events = []
        states_I = [ x for x in states.int2lbl if 'I' in x ]
        for x in states_I:
            # 'I[{i}] -> A[{i}]'
            i = int(x[1:])
            name = 'r_S_{i}'.format(i=i)
            idx = {'i':i}
            rate = params[i]
            ix = [ 'I[{i}]:1'.format(i=i) ]
            jx = [ 'A[{i}]:1'.format(i=i) ]
            e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
            # e = Event( idx=idx, r=r, n='r_S_{i}'.format(i=i), g='Sample', ix=ix, jx=jx )
            events.append(e)

        return events

    # SIRM [m]igrates into new population
    def make_events_m(self, states, params):
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
                rate = params[i][j]
                ix = [ 'I[{i}]:1'.format(i=i) ]
                jx = [ 'I[{j}]:1'.format(j=j) ]
                e = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                # e = Event( idx=idx, r=r, n='r_M_{i}_{j}'.format(i=i, j=j), g='Migrate', ix=ix, jx=jx )
                events.append(e)
        return events
