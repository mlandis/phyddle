import itertools
import numpy as np
from model import Event


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
def GeoSSE_rate_func_x(states, rates):  
    events = []
    for i,x in enumerate(states.int2vec):
        print(i,x)
        if sum(x) == 1:
            for j,y in enumerate(x):
                if y == 1:
                    r = rates[j]
                    xml_str = 'S[{i}] -> X'.format(i=i)
                    idx = {'i':i}
                    e = Event( idx=idx, r=r, n='r_x_{i}'.format(i=i), g='Extinction', x=xml_str, d='' )
                    events.append(e)
    return events


# GeoSSE extirpation rate
def GeoSSE_rate_func_e(states, rates):
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
                xml_str = 'S[{i}] -> S[{i}] + S[{j}]'.format(i=i, j=new_state)
                idx = {'i':i, 'j':i, 'k':new_state}
                e = Event( idx=idx, r=r, n='r_w_{i}_{i}_{j}'.format(i=i, j=new_state), g='Within-region_speciation', x=xml_str, d='' )
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
            xml_str = 'S[{i}] -> S[{j}]'.format(i=i, j=new_state)
            #idx = [i, new_state]
            idx = {'i':i, 'j':new_state}
            e = Event( idx=idx, r=r, n='r_d_{i}_{j}'.format(i=i, j=new_state), g='Dispersal', x=xml_str, d='' )
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
            xml_str = 'S[{i}] -> S[{j}] + S[{k}]'.format(i=i, j=left_state, k=right_state)
            e = Event( idx=idx, r=r, n='r_b_{i}_{j}_{k}'.format(i=i, j=left_state, k=right_state), g='Between-region speciation', x=xml_str, d='' )
            events.append(e)

    # normalize rates
    z = gm(all_rates)
    for i,x in enumerate(events):
        events[i].rate = events[i].rate / z

    return events



# class RateSpace:
#     def __init__(self, rate_space):
#         self.rate_space = rate_space     
#     def make_str(self):
#         # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
#         # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
#         s = 'Ratespace('
#         x = []
#         #for i in self.int2int:
#         # each state in the space is reported as STRING,INT,VECTOR;
#         #    x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
#         s += ';'.join(x) + ')'
#         return s
#     # representation string
#     def __repr__(self):
#         return self.make_str()
#     # print string
#     def __str__(self):
#         return self.make_str()
        
