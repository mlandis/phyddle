#!/usr/local/bin/python3
#from model import *
from model_util import *
from geosse_model_util import *
import string
import itertools

class GeosseModel:
    
    # set up model
    def __init__(self, num_locations, model_variant='equal_rates'):
        
        # create state space
        self.model_type    = 'GeoSSE'
        self.model_variant = model_variant
        self.num_locations = num_locations
        
        # simulation settings
        self.settings = make_settings( self.num_locations )

        # state space
        self.states = make_states( self.num_locations )

        # rate space
        self.rates = make_rates( self.model_variant, self.settings )

        # event space
        self.events = make_events( self.states, self.rates )

        # event space dataframe
        self.df_events = events2df( self.events )

        # state space dataframe
        self.df_states = states2df( self.states )

        # model
        #self.xmlgen = MasterXmlGenerator(self.df_events, self.df_states)
        