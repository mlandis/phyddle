#!/usr/bin/env python
"""
simulate
========
Defines classes and methods for the Simulate step, which generates large numbers
of simulated datasets (in parallel, if desired) that are later formatted and used
to train the neural network.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import os
import subprocess
#import time

# external imports
import numpy as np
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# phyddle imports
from phyddle import utilities

try:
    set_start_method('fork')
except RuntimeError:
    pass

#------------------------------------------------------------------------------#

def load(args): #, mdl=None):
    """
    Load the appropriate simulator.

    This function returns an instance of a simulator class based on the
    `sim_method` key in the provided `args` dictionary. The supported simulators are
    `CommandSimulator` and `MasterSimulator`. If an unsupported value is provided,
    the function returns `None`.

    Args:
        args (dict): A dictionary containing configuration parameters for the
            simulators. Must include the key 'sim_method' which should have a value of
            either 'command' or 'master'.
        mdl (Model): The model instance that the simulator will operate on.

    Returns:
        Simulator: An instance of the `CommandSimulator` or `MasterSimulator` class,
            or `None` if an unsupported `sim_method` is provided.

    """

    # load object
    # sim_method = args['sim_method']
    # if sim_method == 'command':
    return Simulator(args) #, mdl)
    # elif sim_method == 'master':
    #     return MasterSimulator(args) #, mdl)
    # else:
    #     return None

#------------------------------------------------------------------------------#

class Simulator:
    """
    A class representing a simulator.

    Args:
        args (dict): A dictionary containing the simulator arguments.

    Attributes:
        args (dict): The simulator arguments.
        proj (str): The project name.
        verbose (bool): A flag indicating the verbosity level.
        sim_dir (str): The directory for simulation.
        start_idx (int): The start index.
        end_idx (int): The end index.
        num_proc (int): The number of processes.
        use_parallel (bool): A flag indicating whether to use parallel processing.
        sim_logging (bool): A flag indicating whether to enable simulation logging.
        rep_idx (list): A list of replicate indices.
        model (str): The model for simulation.
        logger (utilities.Logger): An instance of the logger for logging.

    """
    def __init__(self, args): #, mdl):
        """
        Initializes a new instance of the Simulator class.

        Args:
            args (dict): A dictionary containing the simulator arguments.
            mdl (str): The model for simulation.

        """
        self.set_args(args)
        self.logger = utilities.Logger(args)
        
        return

    def set_args(self, args):
        """
        Sets the simulator arguments.

        Args:
            args (dict): A dictionary containing the simulator arguments.

        """
        # simulator arguments
        self.args              = args
        self.verbose           = args['verbose']
        self.sim_dir           = args['sim_dir']
        self.sim_proj          = args['sim_proj']
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.num_proc          = args['num_proc']
        self.use_parallel      = args['use_parallel']
        self.sim_command       = args['sim_command']
        self.sim_logging       = args['sim_logging']
        self.rep_idx           = list(range(self.start_idx, self.end_idx))
        self.save_params       = False

        self.sim_proj_dir      = f'{self.sim_dir}/{self.sim_proj}'
        return

    def make_settings_str(self, idx):
        """
        Generates the settings string.

        Args:
            idx (int): The replicate index.

        Returns:
            str: The generated settings string.

        """
        s =  'setting,value\n'
        s += f'sim_proj,{self.sim_proj}\n'
        s += f'sim_command,{self.sim_command}\n'
        s += f'replicate_index,{idx}\n'
        return s

    def run(self):
        """
        This method runs the simulation process.

        Returns:
            list: The result of the simulation process.
        """
        # start run
        utilities.print_step_header('sim', None, self.sim_proj_dir, verbose=self.verbose)

        # prepare workspace
        os.makedirs( self.sim_dir + '/' + self.sim_proj, exist_ok=True )
    
        # dispatch jobs
        utilities.print_str('▪ simulating raw data ...', verbose=self.verbose)
        if self.use_parallel:
            #res = Parallel(n_jobs=self.num_proc)(delayed(self.sim_one)(idx) for idx in tqdm(self.rep_idx))
            #args = [ (idx,) for idx in self.rep_idx ]
            with Pool(processes=self.num_proc) as pool:
                # see https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
                res = list(tqdm(pool.imap(self.sim_one, self.rep_idx, chunksize=1),
                                total=len(self.rep_idx),
                                desc='Simulating'))
                # not needed??
                res = [ x for x in res ]

        else:
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx) ]

        utilities.print_str('... done!', verbose=self.verbose)
        return res

    # main simulation function (looped)
    def sim_one(self, idx):
        """
        This method runs a single simulation iteration.

        Args:
            idx (int): The index of the simulation iteration.

        Returns:
            None
        """
        # get filesystem info for generic job
        out_path   = f'{self.sim_dir}/{self.sim_proj}/sim'
        tmp_fn     = f'{out_path}.{idx}'
        #cmd_log_fn = tmp_fn + '.sim_command.log'

        # run generic job
        # example:
        #   self.sim_command == 'rb sim_one.Rev --args'
        #   tmp_fn == '../workspace/raw_data/Rev_example/sim.0'
        cmd_str = f'{self.sim_command} {tmp_fn}'

        num_attempt = 10
        valid = False
        while not valid and num_attempt > 0:
            try:
                cmd_str_tok = cmd_str.split(' ')
                cmd_out = subprocess.run(cmd_str_tok, capture_output=True)
                #utilities.write_to_file(cmd_out, cmd_log_fn)
                valid = True
            except subprocess.CalledProcessError:
                self.logger.write_log('sim', f'simulation {idx} failed to generate a valid dataset')
                num_attempt -= 1
                valid = False
                #print(f'error for rep_idx={idx}')

        return