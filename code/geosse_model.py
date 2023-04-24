#!/usr/local/bin/python3
from model import *
from model_util import *
import string

#letters   = 'ABCD'
#lbl       = [ ]
#vec       = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1,], [1,1,1] ]
#lbl       = [ 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC' ]

# state space
num_char   = 3
vec        = [ x for x in itertools.product(range(2), repeat=num_char) ][1:]
vec        = sort_binary_vectors(vec)
letters    = string.ascii_uppercase[0:num_char]
lbl        = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
lbl2vec    = { k:v for k,v in list(zip(lbl,vec)) }
statespace = StateSpace(lbl2vec)

# rates
rates = {
    'r_w': sp.stats.expon.rvs(size=num_char),
    'r_e': sp.stats.expon.rvs(size=num_char), 
    'r_d': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char)),
    'r_b': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char))
}
rates['r_x'] = rates['r_e']

# events
events_x = GeoSSE_rate_func_x( statespace, rates['r_x'] )
events_e = GeoSSE_rate_func_e( statespace, rates['r_e'] )
events_w = GeoSSE_rate_func_w( statespace, rates['r_w'] )
events_d = GeoSSE_rate_func_d( statespace, rates['r_d'] )
events_b = GeoSSE_rate_func_b( statespace, rates['r_b'] )
eventspace = events_x + events_e + events_d + events_w + events_b

# event space
df_events = pd.DataFrame({
    'name'     : [ e.name for e in eventspace ],
    'group'    : [ e.group for e in eventspace ], 
    'i'        : [ e.i for e in eventspace ],
    'j'        : [ e.j for e in eventspace ],
    'k'        : [ e.k for e in eventspace ],
    'reaction' : [ e.reaction for e in eventspace ],
    'rate'     : [ e.rate for e in eventspace ]
})

# state space
df_states = pd.DataFrame({
    'lbl' : statespace.int2lbl,
    'int' : statespace.int2int,
    'set' : statespace.int2set,
    'vec' : statespace.int2vec
})

# model
mdl = Model(df_events, df_states)
mdl.make_xml(max_taxa=500, newick_fn='file.nwk', nexus_fn='file.nex', json_fn='file.json')
print(mdl.xml_spec_str)


# simulate