from model_util import Event,States
import itertools
import numpy as np
import scipy as sp


# SIRM simulation settings
def make_settings(num_locations):
    settings = {}

    # generate random starting sizes
    # X ~ Gamma(shape=0.5, scale=1e6) 
    start_sizes = sp.stats.gamma.rvs(size=num_locations, a=0.5, scale=1000000)
    p_start_sizes = start_sizes / np.sum(start_sizes)
    
    #settings['start_state'] = { 'I' : sp.stats.randint.rvs(size=1, low=0, high=num_location)[0] }
    settings['num_locations'] = num_locations
    #settings['model_variant'] = model_variant
    settings['start_state'] = { 'I' : np.random.choice(a=num_locations, size=1, p=p_start_sizes)[0] }
    settings['start_sizes'] = { 'S' : [ int(np.ceil(x)) for x in start_sizes ] }
    settings['sample_population'] = [ 'A' ]
    settings['stop_floor_sizes'] = 0
    settings['stop_ceil_sizes'] = 5000
    settings['rv_fn'] = { 'Infect': sp.stats.expon.rvs,
                          'Migrate': sp.stats.expon.rvs,
                          'Recover': sp.stats.expon.rvs,
                          'Sample': sp.stats.expon.rvs }
    settings['rv_arg'] = { 'Infect': { 'scale' : 0.001 }, # **kwargs
                           'Migrate': { 'scale' : 1. },
                           'Recover': { 'scale' : 0.01 },
                           'Sample': { 'scale' : 0.1 } }
    return settings

# SIRM state space
def make_states(num_locations):
    
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


# SIRM [i]nfection within location
def make_events_i(states, rates):
    events = []
    states_I = [ x for x in states.int2lbl if 'I' in x ]
    for x in states_I:
        i = int(x[1:])
        r = rates[i]
        #xml_str = 'I[{i}] + S[{i}] -> 2 I[{i}]'.format(i=i)
        # 'I[{i}] + S[{i}] -> 2 I[{i}]'
        ix = [ 'I[{i}]:1'.format(i=i), 'S[{i}]:1'.format(i=i) ]
        jx = [ '2I[{i}]:1'.format(i=i) ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_I_{i}'.format(i=i), g='Infect', ix=ix, jx=jx )
        events.append(e)

    return events

# SIRM [r]ecovery within location
def make_events_r(states, rates):
    events = []
    states_I = [ x for x in states.int2lbl if 'I' in x ]
    for x in states_I:
        i = int(x[1:])
        r = rates[i]
        # 'I[{i}] -> R[{i}]'
        ix = [ 'I[{i}]:1'.format(i=i) ]
        jx = [ 'R[{i}]:1'.format(i=i) ]
        #mx = [ ix[0] ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_R_{i}'.format(i=i), g='Recover', ix=ix, jx=jx )
        events.append(e)
        
    return events

# SIRM [s]ampled from infected host
def make_events_s(states, rates):
    events = []
    states_I = [ x for x in states.int2lbl if 'I' in x ]
    for x in states_I:
        i = int(x[1:])
        r = rates[i]
        # xml_str = 'I[{i}] -> A[{i}]'.format(i=i)
        # 'I[{i}] -> A[{i}]'
        ix = [ 'I[{i}]:1'.format(i=i) ]
        jx = [ 'A[{i}]:1'.format(i=i) ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_S_{i}'.format(i=i), g='Sample', ix=ix, jx=jx )
        events.append(e)

    return events

# SIRM [m]igrates into new population
def make_events_m(states, rates):
    events = []
    states_I = [ x for x in states.int2lbl if 'I' in x ]
    state_pairs = list(itertools.product(states_I, states_I))
    for x,y in state_pairs:
        if x != y:
            i = int(x[1:])
            j = int(y[1:])
            r = rates[i][j]
            #xml_str = 'I[{i}] -> I[{j}]'.format(i=i, j=j)
            # 'I[{i}] -> I[{j}]'
            ix = [ 'I[{i}]:1'.format(i=i) ]
            jx = [ 'I[{j}]:1'.format(j=j) ]
            idx = {'i':i, 'j':j}
            e = Event( idx=idx, r=r, n='r_M_{i}_{j}'.format(i=i, j=j), g='Migrate', ix=ix, jx=jx )
            events.append(e)

    return events

def make_events( states, rates ):
    events_i = make_events_i( states, rates['Infect'] )
    events_r = make_events_r( states, rates['Recover'] )
    events_s = make_events_s( states, rates['Sample'] )
    events_m = make_events_m( states, rates['Migrate'] )
    events = events_i + events_m + events_r + events_s
    return events

def make_rates( model_variant, num_locations, settings ):
    rates = {}
    
    # get sim RV functions and arguments
    rv_fn = settings['rv_fn']
    rv_arg = settings['rv_arg']
    #print(rv_fn)
    #print(rv_arg)

    # all rates within an event type are equal, but rate classes are drawn iid
    if model_variant == 'free_rates':
        # check to make sure arguments in settings are applied to model variant
        rates = {
            #'Infect': sp.stats.expon.rvs(size=num_locations),
            #'Recover': sp.stats.expon.rvs(size=num_locations),
            #'Sample': sp.stats.expon.rvs(size=num_locations),
            #'Migrate': sp.stats.expon.rvs(size=num_locations**2).reshape((num_locations,num_locations)),
            'Infect': rv_fn['Infect'](size=num_locations, **rv_arg['Infect']),
            'Recover': rv_fn['Recover'](size=num_locations, **rv_arg['Recover']),
            'Sample': rv_fn['Sample'](size=num_locations, **rv_arg['Sample']),
            'Migrate': rv_fn['Migrate'](size=num_locations**2, **rv_arg['Migrate']).reshape((num_locations,num_locations)),
        }
    # all rates are drawn iid
    elif model_variant == 'equal_rates':
        rates = {
            'Infect': np.full(num_locations, rv_fn['Infect'](size=1, **rv_arg['Infect'])[0]),
            'Recover': np.full(num_locations, rv_fn['Recover'](size=1, **rv_arg['Infect'])[0]),
            'Sample': np.full(num_locations, rv_fn['Sample'](size=1, **rv_arg['Sample'])[0]),
            'Migrate': np.full((num_locations,num_locations), rv_fn['Migrate'](size=1, **rv_arg['Migrate'])[0]),
            # 'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
            # 'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            # 'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            # 'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
        }
    # all rates drawn as log-linear functions of features
    elif model_variant == 'feature_rates':
        # other parameters checked for effect-strength of feature on rates
        rates = {
            'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
            'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
        }
    # return rates
    return rates