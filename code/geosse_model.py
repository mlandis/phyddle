#!/usr/local/bin/python3
from model import Event, StateSpace, RateSpace, Model
import itertools
import string
import scipy as sp
import numpy as np
import pandas as pd
  

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

# GeoSSE extinction rate
def fn_x(states, rates):  
    events = []
    for i,x in enumerate(states.int2vec):
        if sum(x) == 1:
            r = rates[i]
            xml_str = 'S[{i}] -> X'.format(i=i)
            idx = {'i':i}
            e = Event( idx=idx, r=r, n='r_x_{i}'.format(i=i), g='Extinction', x=xml_str, d='' )
            events.append(e)
    return events

# GeoSSE extirpation rate
def fn_e(states, rates):
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
                xml_str = 'S[{i}] -> S[{j}]'.format(i=i, j=new_state)
                idx = {'i':i, 'j':new_state}
                e = Event( idx=idx, r=r, n='r_e_{i}_{j}'.format(i=i, j=new_state), g='Extirpation', x=xml_str, d='' )
                events.append(e)
    return events

# GeoSSE within-region speciation rate
def fn_w(states, rates):
    events = []
    # for each state
    for i,x in enumerate(states.int2vec):
        # for each bit in the state-vector
        for j,y in enumerate(x):
            # if region is occupied
            if y == 1:
                new_state = j
                r = rates[j]
                xml_str = 'S[{i}] -> S[{i}] + S[{j}]'.format(i=i, j=new_state)
                idx = {'i':i, 'j':i, 'k':new_state}
                e = Event( idx=idx, r=r, n='r_w_{i}_{i}_{j}'.format(i=i, j=new_state), g='Within-region_speciation', x=xml_str, d='' )
                events.append(e)
    return events

# GeoSSE dispersal rate
def fn_d(states, rates):
    
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
            xml_str = 'S[{i}] -> S[{j}]'.format(i=i, j=new_state)
            #idx = [i, new_state]
            idx = {'i':i, 'j':new_state}
            e = Event( idx=idx, r=r, n='r_d_{i}_{j}'.format(i=i, j=new_state), g='Dispersal', x=xml_str, d='' )
            events.append(e)

    return events

# GeoSSE between-region speciation rate
def fn_b(states, rates):
    
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
            xml_str = 'S[{i}] -> S[{j}] + S[{k}]'.format(i=i, j=left_state, k=right_state)
            e = Event( idx=idx, r=r, n='r_b_{i}_{j}_{k}'.format(i=i, j=left_state, k=right_state), g='Between-region speciation', x=xml_str, d='' )
            events.append(e)

    # normalize rates
    z = gm(all_rates)
    for i,x in enumerate(events):
        events[i].rate = events[i].rate / z

    return events




# Storing the value in variable result
result = string.ascii_letters

# single event
#evt = []
#evt.append( Event( {'i':0}, 0.3, 'r_e_0', 'extinction') )
#evt.append( Event( {'i':0,'j':1}, 0.4, 'r_d_0_1', 'dispersal') )
#evt.append( Event( {'i':2,'j':0,'k':1}, 0.2, 'r_b_2_0_1', 'between-region speciation') )
#print(evt)

#letters   = 'ABCD'
#lbl       = [ ]
#vec       = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1,], [1,1,1] ]
#lbl       = [ 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC' ]


# state space
num_char  = 3
vec       = [ x for x in itertools.product(range(2), repeat=num_char) ][1:]
vec       = sort_binary_vectors(vec)
letters   = string.ascii_uppercase[0:num_char]
lbl       = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
lbl2vec   = { k:v for k,v in list(zip(lbl,vec)) }
ss = StateSpace(lbl2vec)
#print(ss)

# rate space
#num_char = 3
rates = {
    'r_w': sp.stats.expon.rvs(size=num_char),
    'r_e': sp.stats.expon.rvs(size=num_char), 
    'r_d': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char)),
    'r_b': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char))
}
rates['r_x'] = rates['r_e']
#rs = RateSpace(rates)

# container of rate functions?
# ...

events_x = fn_x( ss, rates['r_x'] )
events_e = fn_e( ss, rates['r_e'] )
events_w = fn_w( ss, rates['r_w'] )
events_d = fn_d( ss, rates['r_d'] )
events_b = fn_b( ss, rates['r_b'] )
events_all = events_x + events_e + events_d + events_w + events_b

#print(events_x)
#print(events_w)
#print(events_e)
#print(events_d)
#print(events_b)
#print(events_all)

# event space
df = pd.DataFrame({
        'name'     : [ e.name for e in events_all ],
        'rate'     : [ e.rate for e in events_all ],
        'group'    : [ e.group for e in events_all ], 
        'reaction' : [ e.reaction for e in events_all ],
        'i'        : [ e.i for e in events_all ],
        'j'        : [ e.j for e in events_all ],
        'k'        : [ e.k for e in events_all ]
    })

print(df)

mdl = Model(df, ss)
mdl.make_xml(max_taxa=500, newick_fn='file.nwk', nexus_fn='file.nex', json_fn='file.json')
