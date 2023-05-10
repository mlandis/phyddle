
import Utilities

class BaseModel:
    def __init__(self, args):
        return
    
    def set_args(self, args):
        self.model_type    = args['model_type']
        self.model_variant = args['model_variant']
        self.rv_fn         = args['rv_fn']
        self.rv_arg        = args['rv_arg']
        return
    
    def set_model(self, seed=None):
        # state space
        self.states      = self.make_states() # self.num_locations )
        # starting population sizes (e.g. SIR models)
        self.start_sizes = self.make_start_sizes( seed )
        # starting state
        self.start_state = self.make_start_state( seed )
        # rate space
        self.rates       = self.make_rates( self.model_variant, seed )
        # event space
        self.events      = self.make_events( self.states, self.rates )
        # event space dataframe
        self.df_events   = Utilities.events2df( self.events )
        # state space dataframe
        self.df_states   = Utilities.states2df( self.states )
        return
    
    def clear_model(self):
        self.is_model_set = False
        self.states = None
        self.rates = None
        self.events = None
        self.df_events = None
        self.df_states = None
    
    def make_settings(self):
        raise NotImplementedError
    
    def make_states(self):
        raise NotImplementedError
    
    def make_events(self):
        raise NotImplementedError
    
    def make_rates(self):
        raise NotImplementedError
    
    def make_start_state(self):
        raise NotImplementedError
    
    def make_start_sizes(self):
        raise NotImplementedError
