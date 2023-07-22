#!/usr/bin/env python
"""
Model
=====
Defines BaseModel class used for internal simulations.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

from phyddle import Utilities
import numpy as np

class BaseModel:
    def __init__(self, args):
        """
        Initializes the BaseModel.

        Args:
            args (dict): A dictionary containing the arguments for initialization.
        """
        return
    
    def set_args(self, args):
        """
        Sets the arguments for the model.

        Args:
            args (dict): A dictionary containing the arguments.
        """
        self.model_type    = args['model_type']
        self.model_variant = args['model_variant']
        self.rv_fn         = args['rv_fn']
        self.rv_arg        = args['rv_arg']
        return
    
    def set_model(self, seed=None):
        """
        Sets the model.

        Args:
            seed (int, optional): The random seed value. Defaults to None.
        """
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
        """
        Clears the model.
        """
        self.is_model_set = False
        self.states = None
        self.params = None
        self.events = None
        self.df_events = None
        self.df_states = None
        return
    
    def make_settings(self):
        """
        Creates the settings for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_states(self):
        """
        Creates the state space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_events(self):
        """
        Creates the event space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_params(self):
        """
        Creates the parameter space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_start_state(self):
        """
        Creates the starting state for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_start_sizes(self):
        """
        Creates the starting sizes for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
