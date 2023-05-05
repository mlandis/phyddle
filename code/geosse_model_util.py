from model_util import Event,States
from model_util import sort_binary_vectors
import model_util
import string
import itertools
import numpy as np
import scipy as sp


def make_settings(num_char):
    settings = {}
    num_states = 2**num_char - 1 
    settings['start_state'] = { 'S' : sp.stats.randint.rvs(size=1, low=0, high=num_states)[0] }    
    settings['rv_fn'] = { 'Within-region speciation': sp.stats.expon,
                          'Extirpation': sp.stats.expon,
                          'Dispersal': sp.stats.expon,
                          'Between-region speciation': sp.stats.expon }
    settings['rv_arg'] = { 'Within-region speciation': { 'scale' : 1. }, # **kwargs
                           'Extirpation': { 'scale' : 1. },
                           'Dispersal': { 'scale' : 1. },
                           'Between-region speciation': { 'scale' : 1. } }
    return settings

def make_states(num_char):
    vec        = [ x for x in itertools.product(range(2), repeat=num_char) ][1:]
    vec        = sort_binary_vectors(vec)
    letters    = string.ascii_uppercase[0:num_char]
    lbl        = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
    lbl2vec    = { k:v for k,v in list(zip(lbl,vec)) }
    states     = States(lbl2vec)
    return states

# GeoSSE extinction rate
def make_events_x(states, rates):  
    events = []
    for i,x in enumerate(states.int2vec):
        if sum(x) == 1:
            for j,y in enumerate(x):
                if y == 1:
                    r = rates[j]
                    xml_str = 'S[{i}]:1 -> X'.format(i=i)
                    idx = {'i':i}
                    e = Event( idx=idx, r=r, n='r_x_{i}'.format(i=i), g='Extinction', x=xml_str )
                    events.append(e)
    return events


# GeoSSE extirpation rate
def make_events_e(states, rates):
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
                r = rates[j]
                xml_str = 'S[{i}]:1 -> S[{j}]:1'.format(i=i, j=new_state)
                idx = {'i':i, 'j':new_state}
                e = Event( idx=idx, r=r, n='r_e_{i}_{j}'.format(i=i, j=new_state), g='Extirpation', x=xml_str )
                events.append(e)
    return events

# GeoSSE within-region speciation rate
def make_events_w(states, rates):
    events = []
    # for each state
    for i,x in enumerate(states.int2vec):
        # for each bit in the state-vector
        for j,y in enumerate(x):
            # if region is occupied
            if y == 1:
                new_state = j
                r = rates[j]
                xml_str = 'S[{i}]:1 -> S[{i}]:1 + S[{j}]:1'.format(i=i, j=new_state)
                idx = {'i':i, 'j':i, 'k':new_state}
                e = Event( idx=idx, r=r, n='r_w_{i}_{i}_{j}'.format(i=i, j=new_state), g='Within-region_speciation', x=xml_str )
                events.append(e)
    return events

# GeoSSE dispersal rate
def make_events_d(states, rates):
    
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
            r = 0.0
            for b,z in enumerate(on[i]):
                r += rates[z][y]
            new_bits = list(states.int2vec[i]).copy()
            new_bits[y] = 1
            #print(states.int2vec[i], '->', new_bits)
            new_state = states.vec2int[ tuple(new_bits) ]
            xml_str = 'S[{i}]:1 -> S[{j}]:1'.format(i=i, j=new_state)
            #idx = [i, new_state]
            idx = {'i':i, 'j':new_state}
            e = Event( idx=idx, r=r, n='r_d_{i}_{j}'.format(i=i, j=new_state), g='Dispersal', x=xml_str )
            events.append(e)

    return events

# GeoSSE between-region speciation rate
def make_events_b(states, rates):
    
    # geometric mean
    def gm(x):
        return np.power( np.prod(x), 1.0/len(x) )

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
            #print('  ',y,z)
            # create event
            left_state = states.set2int[y] 
            right_state = states.set2int[z]
            r = make_split_rate(y,z,rates)
            all_rates.append(r)
            idx = {'i':i, 'j':left_state, 'k':right_state }
            xml_str = 'S[{i}]:1 -> S[{j}]:1 + S[{k}]:1'.format(i=i, j=left_state, k=right_state)
            e = Event( idx=idx, r=r, n='r_b_{i}_{j}_{k}'.format(i=i, j=left_state, k=right_state), g='Between-region speciation', x=xml_str )
            events.append(e)

    # normalize rates
    z = gm(all_rates)
    for i,x in enumerate(events):
        events[i].rate = events[i].rate / z

    return events

def make_events( states, rates ):
    events_x = make_events_x( states, rates['Extinction'] )
    events_e = make_events_e( states, rates['Extirpation'] )
    events_d = make_events_d( states, rates['Dispersal'] )
    events_w = make_events_w( states, rates['Within-region speciation'] )
    events_b = make_events_b( states, rates['Between-region speciation'] )
    events = events_x + events_e + events_d + events_w + events_b
    return events

def make_rates( model_variant, num_char ):
    rates = {}
    
    if model_variant == 'free_rates':
        rates = {
                'Within-region speciation': sp.stats.expon.rvs(size=num_char),
                'Extirpation': sp.stats.expon.rvs(size=num_char), 
                'Dispersal': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char)),
                'Between-region speciation': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char))
            }
        rates['Extinction'] = rates['Extirpation']

    elif model_variant == 'equal_rates':
        rates = {
                'Within-region speciation': np.full(num_char, sp.stats.expon.rvs(size=1)[0]),
                'Extirpation': np.full(num_char, sp.stats.expon.rvs(size=1)[0]), 
                'Dispersal': np.full((num_char,num_char), sp.stats.expon.rvs(size=1)[0]), #sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char)),
                'Between-region speciation': np.full((num_char,num_char), sp.stats.expon.rvs(size=1)[0]) #sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char))
            }
        rates['Extinction'] = rates['Extirpation']

    elif model_variant == 'fig_rates':
        rates = {}
        
    return rates