#!/usr/local/bin/python3
#from model import *
from model_util import *
from sirm_model_util import *
import string
import itertools

class SirmModel:
    
    # set up model
    def __init__(self, num_char, model_variant='equal_rates', feature_set=None):
        
        # create state space
        self.model_type    = 'SIRM'
        self.model_variant = model_variant
        self.num_char      = num_char
     
        # state space
        self.states = make_sirm_states(self.num_char)

        # rate space
        self.rates = make_sirm_rates( self.model_variant, self.num_char )

        # event space
        self.events = make_sirm_events( self.states, self.rates )

        # event space dataframe
        self.df_events = events2df( self.events )

        # state space dataframe
        self.df_states = states2df( self.states )

        # model
        self.xmlgen = MasterXmlGenerator(self.df_events, self.df_states)
