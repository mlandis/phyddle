from model_util import Event,States
import itertools
import numpy as np
import scipy as sp


# SIRM state space
def make_sirm_states(num_location):
    
    # S: Susceptible, I: Infected, R: Recovered
    compartments = 'SIR'
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

    # for each state
    for x in states.int2lbl:
        # get infected compartment in location
        if 'I' in x:
            # get susceptible compartment in location
            i = int(x[1:])
            y = 'S' + x[1:]
            r = rates[i]
            xml_str = 'I[{i}] + S[{i}] -> 2 I[{i}]'.format(i=i)
            idx = {'i':i}
            e = Event( idx=idx, r=r, n='r_I_{i}'.format(i=i), g='Infection', x=xml_str )
            events.append(e)

    return events

# GeoSSE extinction rate
def GeoSSE_rate_func_x(states, rates):  
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



# GeoSSE within-region speciation rate
def GeoSSE_rate_func_w(states, rates):
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
def GeoSSE_rate_func_d(states, rates):
    
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
def GeoSSE_rate_func_b(states, rates):
    
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

def make_sirm_events( states, rates ):
    events_i = make_events_i( states, rates['Infect'] )
    events_m = [] #make_events_m( states, rates['Migrate'] )
    events_r = [] #make_events_r( states, rates['Recover'] )
    events_s = [] #make_events_s( states, rates['Sample'] )

    events = events_i + events_m + events_r + events_s
    return events

def make_sirm_rates( model_variant, num_locations ):
    rates = {}
    
    if model_variant is 'free_rates':
        rates = {
                'Infect': sp.stats.expon.rvs(size=num_locations),
                'Recover': sp.stats.expon.rvs(size=num_locations),
                'Sample': sp.stats.expon.rvs(size=num_locations),
                'Migrate': sp.stats.expon.rvs(size=num_locations**2).reshape((num_locations,num_locations)),
            }

    elif model_variant is 'equal_rates':
        rates = {
                'Infect': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]),
                'Recover': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
                'Sample': np.full(num_locations, sp.stats.expon.rvs(size=1)[0]), 
                'Migrate': np.full((num_locations,num_locations), sp.stats.expon.rvs(size=1)[0])
            }

    return rates