
from phyddle import Utilities
import numpy as np

class BaseModel:
    def __init__(self, args):
        #print('BaseModel')
        return
    
    def set_args(self, args):
        self.model_type    = args['model_type']
        self.model_variant = args['model_variant']
        self.rv_fn         = args['rv_fn']
        self.rv_arg        = args['rv_arg']
        return
    
    def set_model(self, seed=None):
        # set RNG seed if provided
        
        #print("BaseModel.set_model", seed)
        #np.random.seed(seed=seed)
        # set RNG
        self.seed        = seed
        self.rng         = np.random.Generator(np.random.PCG64(seed))
        # state space
        self.states      = self.make_states() # self.num_locations )
        # params space
        self.params      = self.make_params( self.model_variant)
        # starting population sizes (e.g. SIR models)
        # self.start_sizes = self.make_start_sizes()
        # starting state
        self.start_state, self.start_sizes = self.make_start_conditions()
        # event space
        self.events      = self.make_events( self.states, self.params )
        # event space dataframe
        self.df_events   = Utilities.events2df( self.events )
        # state space dataframe
        self.df_states   = Utilities.states2df( self.states )
        return
    
    def clear_model(self):
        self.is_model_set = False
        self.states = None
        self.params = None
        self.events = None
        self.df_events = None
        self.df_states = None
    
    # def get_model_variants(self):
    #     raise NotImplementedError
    
    def make_settings(self):
        raise NotImplementedError
    
    def make_states(self):
        raise NotImplementedError
    
    def make_events(self):
        raise NotImplementedError
    
    def make_params(self):
        raise NotImplementedError
    
    def make_start_state(self):
        raise NotImplementedError
    
    def make_start_sizes(self):
        raise NotImplementedError
    
