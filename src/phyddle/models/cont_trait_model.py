#!/usr/bin/env python
"""
ContTraitModel
===========
Defines a class for a discretized Continuous Trait model.
Dervies from phyddle.Model.BaseModel.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import numpy as np
import itertools

from phyddle.models import model
from phyddle.utilities import States,Event

class ContTraitModel(Model.BaseModel):
    
    # initialize model
    def __init__(self, args):
        super().__init__(args)
        self.set_args(args)
        self.set_model()
        return
    
    # assign initial arguments
    def set_args(self, args):
        super().set_args(args)
        self.num_traits = args['num_traits']
        self.num_bins = args['num_bins']
        if self.num_bins % 2 == 1:
            self.num_bins += 1
        self.bins = np.linspace(-1, 1, num=self.num_bins)
        return
    
    # make model states
    def make_states(self):
        num_bins   = self.num_bins
        num_traits = self.num_traits
        bins       = self.bins
        # get range of idx for trait ID and bins for trait values
        #trait_idx     = list(range(num_traits))
        bin_idx       = [ int(x - num_bins/2) for x in list(range(num_bins+1)) ]
        vec           = list(itertools.combinations_with_replacement(bin_idx,num_traits))
        #trait_bin_idx = list(itertools.product(trait_idx,bin_idx))
        # convert to state space
        # I guess we want K states with ordered bin values
        # Labels: X_-3_-3_-3, X_-3_-3_-2, ..., X_3_3_3
        lbl           = [ 'X_' + '_'.join( [ str(y) for y in x ] ) for x in vec ]
        # Vectors: [-3,-3,-3], [-3,-3,-2], ..., [0,0,0], ..., [3,3,3]
        #vec           = [ [x] for x in bin_idx ]
        lbl2vec       = { k:v for k,v in list(zip(lbl,vec)) }
        states        = States(lbl2vec)
        return states

    # make starting state for simulation
    def make_start_state(self):
        # { 'S' : 0 }
        rv_fn = self.rv_fn
        rv_arg = self.rv_arg
        num_traits = self.num_traits
        start_states = rv_fn['mu'](size=num_traits, random_state=self.rng, **rv_arg['mu'])
        # find this one in the state space
        ret = { 'S' : start_states }
        return ret
    
    # make starting sizes for compartments
    def make_start_sizes(self):
        return {}

    # get all model rates
    def make_params(self, model_variant):
        params = {}
        
        # get settings
        num_traits = self.num_traits
        rv_fn = self.rv_fn
        rv_arg = self.rv_arg

        # build rates
        if model_variant == 'BM_iid':
            # X_i ~ BM(sigma_i)
            params = {
                    'mu': rv_fn['mu'](size=num_traits, random_state=self.rng, **rv_arg['mu']),
                    'sigma': rv_fn['sigma'](size=num_traits, random_state=self.rng, **rv_arg['sigma'])
                }
        
        if model_variant == 'OU_iid':
            # X_i ~ OU(alpha_i, theta_i, sigma_i)
            params = {
                    'alpha': rv_fn['alpha'](size=num_traits, random_state=self.rng, **rv_arg['alpha']),
                    'theta': rv_fn['theta'](size=num_traits, random_state=self.rng, **rv_arg['theta']),
                    'sigma': rv_fn['sigma'](size=num_traits, random_state=self.rng, **rv_arg['sigma']),
                }

        if model_variant == 'OU_sum_OU_iid':
            # Z = sum(X)
            # Z ~ OU(alpha^z, theta^z, sigma^z)
            # X_i ~ OU(alpha_i^x, theta_i^x, sigma_i^x)
            params = {
                    'alpha_z' : rv_fn['alpha_z'](size=1, random_state=self.rng, **rv_arg['alpha_z']),
                    'theta_z' : rv_fn['theta_z'](size=1, random_state=self.rng, **rv_arg['theta_z']),
                    'sigma_z' : rv_fn['sigma_z'](size=1, random_state=self.rng, **rv_arg['sigma_z']),
                    'alpha_x' : rv_fn['alpha_x'](size=num_traits, random_state=self.rng, **rv_arg['alpha_x']),
                    'theta_x' : rv_fn['theta_x'](size=num_traits, random_state=self.rng, **rv_arg['theta_x']),
                    'sigma_x' : rv_fn['sigma_x'](size=num_traits, random_state=self.rng, **rv_arg['sigma_x']),
                }

        return params
    
    # simple 1-layer BM or OU
    def make_events_diffusion_1L(self, states, rates):
        events = []
        num_traits = self.num_traits
        bins = self.bins

        # setup rates
        theta = np.repeat(0.0, num_traits)
        if 'theta' in rates:
            theta = rates['theta']
        
        alpha = np.repeat(0.0, num_traits)
        if 'alpha' in rates:
            alpha = rates['alpha']

        # setup events
        for k in num_traits:
            for i,x in enumerate(states.int2vec):

                # OU pseudo-deterministic change in dx
                dt = 0.1
                det_change = alpha[k] * (theta[k] - bins[i])
                j_pull = i + np.sign(det_change)
                name = 'pull_dx_{i}'.format(i=i)
                idx = {'i':i, 'j':j_pull}
                rate = det_change
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=j_pull) ]
                e = Event( g='Pull', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)

                # BM stochastic decrease dx
                name = 'decrease_dx_{i}'.format(i=i)
                idx = {'i':i}
                rate = rates[i]
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=i-1) ]
                e = Event( g='Diffusion', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)

                # BM stochastic increase dx
                name = 'increase_dx_{i}'.format(i=i)
                idx = {'i':i}
                rate = rates[i]
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=i+1) ]
                e = Event( g='Diffusion', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)
        return events

    # simple 1-layer BM or OU
    def make_events_diffusion_2L(self, states, rates):
        events = []
        num_traits = self.num_traits
        bins = self.bins

       # setup rates
        theta = np.repeat(0.0, num_traits)
        if 'theta_z' in rates:
            theta = rates['theta_z']
        alpha = np.repeat(0.0, num_traits)
        if 'alpha_z' in rates:
            alpha = rates['alpha_z']

        # setup events
        for k in num_traits:
            for i,x in enumerate(states.int2vec):

                # OU pseudo-deterministic change in dx
                dt = 0.1
                det_change = alpha[k] * (theta[k] - bins[i])
                j_pull = i + np.sign(det_change)
                name = 'pull_dx_{i}'.format(i=i)
                idx = {'i':i, 'j':j_pull}
                rate = det_change
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=j_pull) ]
                e = Event( g='Pull', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)

                # BM stochastic decrease dx
                name = 'decrease_dx_{i}'.format(i=i)
                idx = {'i':i}
                rate = rates[i]
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=i-1) ]
                e = Event( g='Diffusion', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)

                # BM stochastic increase dx
                name = 'increase_dx_{i}'.format(i=i)
                idx = {'i':i}
                rate = rates[i]
                ix = [ 'X[{i}]:1'.format(i=i) ]
                jx = [ 'X[{i}]:1'.format(i=i+1) ]
                e = Event( g='Diffusion', n=name, idx=idx, r=rate, ix=ix, jx=jx )
                events.append(e)
        return events

    # make list of all events in model
    def make_events(self, states, rates):
        if self.model_variant == 'BM_iid':
            events = self.make_events_diffusion_1L( states, rates )
            return events
        elif self.model_variant == 'OU_iid':
            events = self.make_events_diffusion_1L( states, rates )
            return events

    
