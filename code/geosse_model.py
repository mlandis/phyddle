#!/usr/local/bin/python3
#from model import *
from model_util import *
from geosse_model_util import *
import string
import itertools

class GeosseModel:
    
    # set up model
    def __init__(self, num_char):
        
        # create state space
        self.model_type = 'GeoSSE'
        self.num_char   = num_char
        self.vec        = [ x for x in itertools.product(range(2), repeat=self.num_char) ][1:]
        self.vec        = sort_binary_vectors(self.vec)
        self.letters    = string.ascii_uppercase[0:self.num_char]
        self.lbl        = [ ''.join([ self.letters[i] for i,y in enumerate(x) if y == 1 ]) for x in self.vec ]
        self.lbl2vec    = { k:v for k,v in list(zip(self.lbl,self.vec)) }
        self.states    = States(self.lbl2vec)

        # rate space
        self.rates = {
            'Within-region speciation': sp.stats.expon.rvs(size=self.num_char),
            'Extirpation': sp.stats.expon.rvs(size=self.num_char), 
            'Dispersal': sp.stats.expon.rvs(size=self.num_char**2).reshape((self.num_char,self.num_char)),
            'Between-region speciation': sp.stats.expon.rvs(size=self.num_char**2).reshape((self.num_char,self.num_char))
        }
        self.rates['Extinction'] = self.rates['Extirpation']

        # event space
        self.events = make_geosse_events( self.states, self.rates )

        # event space dataframe
        self.df_events = events2df( self.events )

        # state space dataframe
        self.df_states = states2df( self.states )

        # model
        self.xmlgen = MasterXmlGenerator(self.df_events, self.df_states)

    # should model handle various state/file conversions?

# num_char   = 3
# vec        = [ x for x in itertools.product(range(2), repeat=num_char) ][1:]
# vec        = sort_binary_vectors(vec)
# letters    = string.ascii_uppercase[0:num_char]
# lbl        = [ ''.join([ letters[i] for i,y in enumerate(x) if y == 1 ]) for x in vec ]
# lbl2vec    = { k:v for k,v in list(zip(lbl,vec)) }
# states    = States(lbl2vec)

# # rate space
# rates = {
#     'Within-region speciation': sp.stats.expon.rvs(size=num_char),
#     'Extirpation': sp.stats.expon.rvs(size=num_char), 
#     'Dispersal': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char)),
#     'Between-region speciation': sp.stats.expon.rvs(size=num_char**2).reshape((num_char,num_char))
# }
# rates['Extinction'] = rates['Extirpation']

# # event space
# events = make_geosse_events( states, rates )

# # event space
# df_events = events2df( events )

# # state space
# df_states = states2df( states )

# # model
# mdl = Model(df_events, df_states)
# mdl.make_xml(max_taxa=500, newick_fn='file.nwk', nexus_fn='file.nex', json_fn='file.json')
# print(mdl.xml_spec_str)