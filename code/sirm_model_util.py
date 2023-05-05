from model_util import Event,States
import itertools
import numpy as np
import scipy as sp


# SIRM state space
def make_sirm_states(num_location):
    
    # S: Susceptible, I: Infected, R: Recovered, A: Acquired ('Sampled')
    compartments = 'SIRA'
    # Location 0, Location 1, Location 2, ...
    locations = [ str(x) for x in list(range(num_location)) ]
    # S0, S1, S2, ..., I0, I1, I2, ..., R0, R1, R2, ...
    lbl = [ ''.join(x) for x in list(itertools.product(compartments,locations)) ]
    # 1000, 0100, 0010, 0001 (one-hot encoding for N locations)
    vec = np.identity(num_location, dtype='int').tolist()
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
        ix = [ 'I[{i}]'.format(i=i), 'S[{i}]'.format(i=i) ]
        jx = [ '2S[{i}]'.format(i=i) ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_I_{i}'.format(i=i), g='Infection', x='', ix=ix, jx=jx )
        events.append(e)

    return events

# SIRM [r]ecovery within location
def make_events_r(states, rates):
    events = []
    states_I = [ x for x in states.int2lbl if 'I' in x ]
    for x in states_I:
        i = int(x[1:])
        r = rates[i]
        #xml_str = 'I[{i}] -> R[{i}]'.format(i=i)
        # 'I[{i}] -> R[{i}]'
        ix = [ 'I[{i}]'.format(i=i) ]
        jx = [ 'R[{i}]'.format(i=i) ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_R_{i}'.format(i=i), g='Recover', x='', ix=ix, jx=jx )
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
        ix = [ 'I[{i}]'.format(i=i) ]
        jx = [ 'A[{i}]'.format(i=i) ]
        idx = {'i':i}
        e = Event( idx=idx, r=r, n='r_S_{i}'.format(i=i), g='Sample', x='', ix=ix, jx=jx )
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
            ix = [ 'I[{i}]'.format(i=i) ]
            jx = [ 'A[{j}]'.format(j=j) ]
            idx = {'i':i, 'j':j}
            e = Event( idx=idx, r=r, n='r_M_{i}_{j}'.format(i=i, j=j), g='Migrate', x='', ix=ix, jx=jx )
            events.append(e)

    return events

def make_sirm_events( states, rates ):
    events_i = make_events_i( states, rates['Infect'] )
    events_r = make_events_r( states, rates['Recover'] )
    events_s = make_events_s( states, rates['Sample'] )
    events_m = make_events_m( states, rates['Migrate'] )
    events = events_i + events_m + events_r + events_s
    return events

def make_sirm_rates( model_variant, num_locations ):
    rates = {}
    # all rates within an event type are equal, but rate classes are drawn iid
    if model_variant is 'free_rates':
        rates = {
            'Infect': sp.stats.expon.rvs(size=num_locations),
            'Recover': sp.stats.expon.rvs(size=num_locations),
            'Sample': sp.stats.expon.rvs(size=num_locations),
            'Migrate': sp.stats.expon.rvs(size=num_locations**2).reshape((num_locations,num_locations)),
        }
    # all rates are drawn iid
    elif model_variant is 'equal_rates':
        rates = {
            'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
            'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
        }
    # all rates drawn as log-linear functions of features
    elif model_variant is 'feature_rates':
        rates = {
            'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
            'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
            'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
        }
    # return rates
    return rates