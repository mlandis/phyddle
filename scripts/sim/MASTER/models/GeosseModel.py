#!/usr/bin/env python
"""
GeosseModel
===========
Defines a class for a Geographic State-Dependent Speciation-Extinction (GeoSSE) model.
Dervies from phyddle.Model.BaseModel.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import string
import itertools
import numpy as np
import scipy as sp
import pandas as pd

from models.model import BaseModel
from master_util import sort_binary_vectors,States,Event

variant_registry = []
variant_registry_names = ['variant_name',      'description'] 
variant_registry.append( ['free_rates',        'rates differ among all events within type'] )
variant_registry.append( ['equal_rates',       'rates equal among all events within type'] )
variant_registry.append( ['density_effect',    'equal_rates + local density-dependent extinction'] )
variant_registry = pd.DataFrame( variant_registry, columns = variant_registry_names)

class GeosseModel(BaseModel):
    
    # initialize model
    def __init__(self, args):
        super().__init__(args)
        self.set_args(args)
        self.set_model()
        return
    
    # assign initial arguments
    def set_args(self, args):
        super().set_args(args)
        self.num_char = args['num_char']
        self.num_ranges = 2**self.num_char - 1
        return

    # make model states
    def make_states(self):
        num_char = self.num_char
        vec        = [ x for x in itertools.product(range(2), repeat=num_char) ][1:]
        vec        = sort_binary_vectors(vec)
        letters    = string.ascii_uppercase[0:num_char]
        lbl        = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
        lbl2vec    = { k:v for k,v in list(zip(lbl,vec)) }
        states     = States(lbl2vec)

        self.make_region_lookup_dict(states)

        return states

    # make dictionary containing all states that contain a given region
    def make_region_lookup_dict(self, states):
        
        self.region_lookup_dict = {}
        
        # for each region, collect indexes for all ranges that contain that region
        # example with 3-region system:
        #   0: 100; 1: 010; 2: 001; 3:110; 4:101; 5:011; 6:111
        # gives us
        #   x[0] = [ 0, 3, 4, 6 ]
        #   x[1] = [ 1, 3, 5, 6 ]
        #   x[2] = [ 2, 4, 5, 6 ]
        for region_idx in range(self.num_char):
            self.region_lookup_dict[region_idx] = []

        # visit each range vector
        for range_state,range_vector in enumerate(states.int2vec):
            # visit each region
            for region_idx,region_state in enumerate(range_vector):
                # if region is occupied, add the range-state to the dictionary
                if region_state == 1:
                    self.region_lookup_dict[region_idx].append(range_state)

        return
    
    # make starting state for simulation
    def make_start_conditions(self):
        # { 'S' : 0 }
        start_state = {}
        start_sizes = {}

        # get starting state for lineage
        num_ranges = self.num_ranges
        idx = sp.stats.randint.rvs(low=0, high=num_ranges, size=1, random_state=self.rng)[0]
        start_state['S'] = idx

        # get starting regions of lineage
        start_range_vec = self.states.int2vec[idx]
        start_sizes['G'] = start_range_vec
        # for i,j in enumerate(start_range_vec):
        #     if j == 1:
        #         start_sizes['G'].append( i )

        return start_state, start_sizes
    
    # make starting sizes for compartments
    # def make_start_sizes(self):
        # return {}

    # get all model rates
    def make_params(self):
        params = {}
        
        # get settings
        num_char = self.num_char
        rv_fn = self.rv_fn
        rv_arg = self.rv_arg
        model_variant = self.model_variant

        # build rates
        if model_variant == 'free_rates':
            params = {
                    'w': rv_fn['w'](size=num_char, random_state=self.rng, **rv_arg['w']),
                    'e': rv_fn['e'](size=num_char, random_state=self.rng, **rv_arg['e']),
                    'd': rv_fn['d'](size=num_char**2, random_state=self.rng, **rv_arg['d']).reshape((num_char,num_char)),
                    'b': rv_fn['b'](size=num_char**2, random_state=self.rng, **rv_arg['b']).reshape((num_char,num_char))
                }
            params['x'] = params['x']

        elif model_variant == 'equal_rates':
            params = {
                    'w': np.full(num_char, rv_fn['w'](size=1, random_state=self.rng, **rv_arg['w'])[0]),
                    'e': np.full(num_char, rv_fn['e'](size=1, random_state=self.rng, **rv_arg['e'])[0]),
                    'd': np.full((num_char,num_char), rv_fn['d'](size=1, random_state=self.rng, **rv_arg['d'])[0]),
                    'b': np.full((num_char,num_char), rv_fn['b'](size=1, random_state=self.rng, **rv_arg['b'])[0])
                }
            params['x'] = params['e']

        elif model_variant == 'fig_rates':
            params = {}
        
        elif model_variant == 'density_effect':
            params = {
                    'w': np.full(num_char, rv_fn['w'](size=1, random_state=self.rng, **rv_arg['w'])[0]),
                    'e': np.full(num_char, rv_fn['e'](size=1, random_state=self.rng, **rv_arg['e'])[0]),
                    'd': np.full((num_char,num_char), rv_fn['d'](size=1, random_state=self.rng, **rv_arg['d'])[0]),
                    'b': np.full((num_char,num_char), rv_fn['b'](size=1, random_state=self.rng, **rv_arg['b'])[0]),
                    'ed': np.full(num_char, rv_fn['ed'](size=1, random_state=self.rng, **rv_arg['ed'])[0])
                }
            # params = {
            #         'w': rv_fn['w'](size=num_char, random_state=self.rng, **rv_arg['w']),
            #         'e': rv_fn['e'](size=num_char, random_state=self.rng, **rv_arg['e']),
            #         'd': np.full((num_char,num_char), rv_fn['d'](size=1, random_state=self.rng, **rv_arg['d'])[0]),
            #         'b': np.full((num_char,num_char), rv_fn['b'](size=1, random_state=self.rng, **rv_arg['b'])[0]),
            #         'ed': rv_fn['ed'](size=num_char, random_state=self.rng, **rv_arg['ed']),
            #     }
            params['x'] = params['e']
            params['xd'] = params['ed']

        return params
    
    # make list of all events in model
    def make_events(self, states, rates):
        
        # lineage extinction events
        events_x = self.make_events_x( states, rates['x'] )
        
        # regional extinction (extirpation) events
        events_e = self.make_events_e( states, rates['e'] )
        
        # dispersal events
        events_d = self.make_events_d( states, rates['d'] )
        
        # within-region speciation events
        events_w = self.make_events_w( states, rates['w'] )
        
        # between-region speciation events
        events_b = self.make_events_b( states, rates['b'] )
        
        core_events = events_w + events_d + events_b + events_x + events_e

        extra_events = []
        if self.model_variant == 'density_effect':
             extra_events_x_DE = self.make_events_x_DE( states, rates['xd'] )
             extra_events_e_DE = self.make_events_e_DE( states, rates['ed'] )
             extra_events = extra_events + extra_events_x_DE + extra_events_e_DE

        events = core_events + extra_events

        return events

    # GeoSSE lineage extinction rate
    def make_events_x(self, states, rates):  
        group = 'Extinction'
        events = []
        for range_state,range_vector in enumerate(states.int2vec):
            if sum(range_vector) == 1:
                for region_idx,region_state in enumerate(range_vector):
                    if region_state == 1:
                        name = f'r_x_{range_state}'
                        idx  = {'i': range_state}
                        rate = rates[region_idx]
                        ix   = [ f'S[{range_state}]:1' ]
                        jx   = [ f'X[{region_idx}]', f'L[{region_idx}]' ]
                        e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                        events.append(e)
        return events

    # GeoSSE lineage extinction rate due to density effects
    def make_events_x_DE(self, states, rates):  
        group = 'Extinction_DE'
        events = []
        for range_state,range_vector in enumerate(states.int2vec):
            if sum(range_vector) == 1:
                for region_idx,region_state in enumerate(range_vector):
                    if region_state == 1:
                        name = 'r_xd_{i}'.format(i=range_state)
                        idx  = {'i': range_state}
                        rate = rates[region_idx]
                        # enumerate over all states that contain region j
                        for other_range_state in self.region_lookup_dict[region_idx]:
                            ix = [ f'S[{other_range_state}]', f'S[{range_state}]:1' ]
                            jx = [ f'S[{other_range_state}]', f'X[{region_idx}]', f'L[{region_idx}]' ]
                            e  = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                            events.append(e)
        return events
    
    # GeoSSE regional extinction (extirpation) rate 
    def make_events_e(self, states, rates):
        group = 'Extirpation'
        events = []
        for range_state,range_vector in enumerate(states.int2vec):
            for region_idx,region_state in enumerate(range_vector):
                if sum(range_vector) >= 2 and region_state == 1:
                    new_bits = list(range_vector).copy()
                    new_bits[region_idx] = 0
                    new_bits = tuple(new_bits)
                    new_state = states.vec2int[ new_bits ]
                    name = f'e_{range_state}_{new_state}'
                    rate = rates[region_idx]
                    idx  = {'i':range_state, 'j':new_state}
                    ix   = [ f'S[{range_state}]:1' ]
                    jx   = [ f'S[{new_state}]:1', f'L[{region_idx}]' ]
                    e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                    events.append(e)
        return events

    # GeoSSE regional extinction (extirpation) rate due to density effects
    def make_events_e_DE(self, states, rates):
        group = 'Extirpation_DE'
        events = []
        for range_state,range_vector in enumerate(states.int2vec):
            for region_idx,region_state in enumerate(range_vector):
                if sum(range_vector) >= 2 and region_state == 1:
                    new_bits = list(range_vector).copy()
                    new_bits[region_idx] = 0
                    new_bits = tuple(new_bits)
                    new_state = states.vec2int[ new_bits ]
                    name = f'e_{range_state}_{new_state}'
                    rate = rates[region_idx]
                    # enumerate over all states that contain region_idx
                    for other_range_state in self.region_lookup_dict[region_idx]:
                        idx = {'i':range_state, 'j':new_state}
                        ix  = [ f'S[{other_range_state}]', f'S[{range_state}]:1' ]
                        jx  = [ f'S[{other_range_state}]', f'S[{new_state}]:1', f'L[{region_idx}]' ]
                        e   = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                        events.append(e)
        return events

    # GeoSSE within-region speciation rate
    def make_events_w(self, states, rates):
        group = 'Within-region speciation'
        events = []
        for range_state,range_vector in enumerate(states.int2vec):
            # for each bit in the state-vector
            for region_idx,region_state in enumerate(range_vector):
                # if region is occupied
                if region_state == 1:
                    name = f'w_{range_state}_{range_state}_{region_idx}'
                    rate = rates[region_idx]
                    idx  = {'i': range_state, 'j': range_state, 'k': region_idx}
                    ix   = [ f'S[{range_state}]:1' ]
                    jx   = [ f'S[{range_state}]:1', f'S[{region_idx}]:1', f'G[{region_idx}]' ]
                    e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
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
        for range_state,range_vector in enumerate(states.int2vec):
            on.append( [] )
            off.append( [] )
            for region_idx,region_state in enumerate(range_vector):
                if region_state == 0:
                    off[range_state].append(region_idx)
                else:
                    on[range_state].append(region_idx)
        # convert to dispersal events
        events = []
        for range_state in range(num_states):
            if sum(on[range_state]) == 0:
                next
            for a,new_region_idx in enumerate(off[range_state]):
                rate = 0.0
                for b,curr_region_idx in enumerate(on[range_state]):
                    # sum of dispersal rates into off_region_idx
                    rate += rates[curr_region_idx][new_region_idx]
                new_bits = list(states.int2vec[range_state]).copy()
                new_bits[new_region_idx] = 1
                new_state = states.vec2int[ tuple(new_bits) ]
                name = f'd_{range_state}_{new_state}'
                idx  = {'i':range_state, 'j':new_state}
                ix   = [ f'S[{range_state}]:1' ]
                jx   = [ f'S[{new_state}]:1', f'G[{new_region_idx}]' ]
                e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)
        return events
    
    # # GeoSSE dispersal rate w/ Density Effects
    # def make_events_d(self, states, rates):
    #     group = 'Dispersal_DE'
    #     # how many compound states
    #     num_states = len(states.int2int)
    #     # generate on/off bits
    #     on  = []
    #     off = []
    #     for range_state,range_vector in enumerate(states.int2vec):
    #         on.append( [] )
    #         off.append( [] )
    #         for region_idx,region_state in enumerate(range_vector):
    #             if region_state == 0:
    #                 off[range_state].append(region_idx)
    #             else:
    #                 on[range_state].append(region_idx)
    #     # convert to dispersal events
    #     events = []
    #     for range_state in range(num_states):
    #         if sum(on[range_state]) == 0:
    #             next
    #         for a,new_region_idx in enumerate(off[range_state]):
    #             rate = 0.0
    #             for b,curr_region_idx in enumerate(on[range_state]):
    #                 # sum of dispersal rates into off_region_idx
    #                 rate += rates[curr_region_idx][new_region_idx]
    #             new_bits = list(states.int2vec[range_state]).copy()
    #             new_bits[new_region_idx] = 1
    #             new_state = states.vec2int[ tuple(new_bits) ]
    #             name = f'd_{range_state}_{new_state}'
    #             idx  = {'i':range_state, 'j':new_state}
    #             ix   = [ f'S[{range_state}]:1' ]
    #             jx   = [ f'S[{new_state}]:1', f'G[{new_region_idx}]' ]
    #             e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
    #             events.append(e)
    #     return events

    # GeoSSE between-region speciation rate
    def make_events_b(self, states, rates, normalize_rates=False):
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
        
        # compute split rates of whole into left and right parts
        def make_split_rate(left, right, rates):
            r = 0.0
            for i in left:
                for j in right:
                    r += 1.0 / rates[i][j]
            return 1.0 / r
                
        # get unique nonempty left/right splits for each range
        events = []
        all_rates = []
        for range_state,anc_region_set in enumerate(states.int2set):
            # get all non-empty splits
            splits = list(powerset(anc_region_set))[1:-1]
            # for each right/left range-state outcome
            for left_region_set in splits:
                right_region_set = tuple(set(anc_region_set).difference(left_region_set))
                left_state  = states.set2int[left_region_set] 
                right_state = states.set2int[right_region_set]
                # build event
                name = f'b_{range_state}_{left_state}_{right_state}'
                rate = make_split_rate(left_region_set,right_region_set,rates)
                idx  = {'i':range_state, 'j':left_state, 'k':right_state }
                ix   = [ f'S[{range_state}]:1' ]
                jx   = [ f'S[{left_state}]:1', f'S[{right_state}]:1' ]
                e    = Event( g=group, n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)
                all_rates.append(rate)
        
        # normalize rates
        if normalize_rates:
            z = gm(all_rates)
            for range_state,range_vector in enumerate(events):
               events[range_state].rate = events[range_state].rate / z

        return events


    
