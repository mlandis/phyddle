#!/usr/bin/env python
"""
simulate
========
Defines classes and methods for the Simulate step, which generates large numbers
of simulated datasets (in parallel, if desired) that are later formatted and
used to train the neural network.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import os
import subprocess

# external imports
from multiprocessing import Pool, set_start_method

from tqdm import tqdm

# phyddle imports
from phyddle import utilities as util

# Uncomment to debug multiprocessing
# import multiprocessing.util as mp_util
# mp_util.log_to_stderr(mp_util.SUBDEBUG)

# Allows multiprocessing fork (not spawn) new processes on Unix-based OS.
# However, import phyddle.format also calls set_start_method('fork'), and the
# function throws RuntimeError when called the 2nd+ time within a single Python.
# We handle this with a try-except block.
try:
    set_start_method('fork')
except RuntimeError:
    pass

#------------------------------------------------------------------------------#

def load(args):
    """
    Load a Simulator object.

    This function creates an instance of the Simulator class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.
    """

    # load object
    simulate_method = 'default'
    if simulate_method == 'default':
        return Simulator(args)
    else:
        return NotImplementedError

#------------------------------------------------------------------------------#

class Simulator:
    """
    Class for simulating phylogenetic datasets that can be converted into
    tensors with the Format step.
    """
    def __init__(self, args):
        """
        Initializes a new Simulator object.

        Args:
            args (dict): Contains phyddle settings.
        """
        # initialize with phyddle settings
        self.set_args(args)
        # directory to store simulations
        self.sim_proj_dir = f'{self.sim_dir}/{self.sim_proj}'
        # run() attempts to generate one simulation per value in rep_idx,
        # where rep_idx is list of unique ints to identify simulated datasets
        self.rep_idx = list(range(self.start_idx, self.end_idx))
        # create logger to track runtime info
        self.logger = util.Logger(args)
        #done
        return

    def set_args(self, args):
        """
        Assigns phyddle settings as Simulator attributes.

        Args:
            args (dict): Contains phyddle settings.
        """
        # simulator arguments
        self.args         = args
        self.verbose      = args['verbose']
        self.sim_dir      = args['sim_dir']
        self.sim_proj     = args['sim_proj']
        self.start_idx    = args['start_idx']
        self.end_idx      = args['end_idx']
        self.num_proc     = args['num_proc']
        self.use_parallel = args['use_parallel']
        self.sim_command  = args['sim_command']
        self.sim_logging  = args['sim_logging']
        # TODO: automatic set arg
        # step_args = util.make_step_args('S', args)
        # for k,v in step_args.items():
        #     setattr(self, k, v)
        return

    def run(self):
        """
        Simulates training examples.

        This creates the target directory for new simulations then runs all
        simulation jobs.
        
        Simulation jobs are numbered by the replicate-index list (self.rep_idx). 
        Each job is executed by calling self.sim_one(idx) where idx is a unique
        value in self.rep_idx.
 
        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.
        """
        verbose = self.verbose

        # print header
        util.print_step_header('sim', None, self.sim_proj_dir, verbose)

        # prepare workspace
        os.makedirs(self.sim_proj_dir, exist_ok=True)
    
        # dispatch jobs
        util.print_str('▪ Simulating raw data ...', verbose)
        if self.use_parallel:
            # parallel jobs
            # Note, it's critical to call this as list(tqdm(pool.imap(...)))
            # - pool.imap runs the parallelization
            # - tqdm generates progress bar
            # - list acts as a finalizer for the pool.imap work
            # Also, no major performance difference between
            # - imap vs imap_unordered
            # - chunksize=1 vs chunksize=5
            # - worth testing more, though
            with Pool(processes=self.num_proc) as pool:
                 res = list(tqdm(pool.imap(self.sim_one, self.rep_idx, chunksize=1),
                            total=len(self.rep_idx),
                            desc='Simulating'))
            
        else:
            # serial jobs
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx,
                                                      total=len(self.rep_idx),
                                                      desc='Simulating') ]

        # done
        util.print_str('... done!', verbose)
        return
    
    # main simulation function (looped)
    def sim_one(self, idx):
        """
        Executes a single simulation.

        This method uses subprocess to run a simulation program. First,
        it constructs a command string (cmd_str) from the basic command
        (stored in self.sim_command) and the filepath prefix for an argument
        (stored in tmp_fn).

        For example

            self.sim_command == 'rb sim_one.Rev --args'
            tmp_fn           == '../my_sim_dir/my_proj/sim.0'

        yields

            cmd_str == 'rb sim_one.Rev --args ../my_sim_dir/my_proj/sim.0'
            
        where the script sim_one.Rev is expected to generate training examples,
        such as:

            sim.0.tre
            sim.0.dat.nex
            sim.0.labels.csv

        Args:
            idx (int): The index of the simulation iteration.
        """
        # get filesystem info for generic job
        out_path   = f'{self.sim_dir}/{self.sim_proj}/sim'
        tmp_fn     = f'{out_path}.{idx}'
        cmd_str    = f'{self.sim_command} {tmp_fn}'
        stdout_fn  = f'{tmp_fn}.stdout.log'
        stderr_fn  = f'{tmp_fn}.stderr.log'

        # run generic job
        num_attempt = 10
        valid = False
        while not valid and num_attempt > 0:
            try:
                # create command tokens
                cmd_str_tok = cmd_str.split(' ')
                # run command
                cmd_res = subprocess.run(cmd_str_tok, capture_output=True)
                # save stdout
                cmd_stdout = cmd_res.stdout.decode('UTF-8')
                util.write_to_file(cmd_stdout, stdout_fn)
                # save stderr
                cmd_stderr = cmd_res.stderr.decode('UTF-8')
                if cmd_stderr != '':
                    util.write_to_file(cmd_stderr, stderr_fn)
                # done simulating
                valid = True
            except subprocess.CalledProcessError:
                error_msg = 'generated a non-zero exit code.'
                self.logger.write_log('sim', f'Simulation {idx}: {error_msg}')
                num_attempt -= 1
                valid = False

        return
    
#------------------------------------------------------------------------------#