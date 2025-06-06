#!/usr/bin/env python
"""
simulate
========
Defines classes and methods for the Simulate step, which generates large numbers
of simulated datasets (in parallel, if desired) that are later formatted and
used to train the neural network.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import sys
import os
import shutil
import subprocess

# external imports
from multiprocessing import Pool, set_start_method, cpu_count

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

##################################################


def load(args):
    """Load a Simulator object.

    This function creates an instance of the Simulator class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    Returns:
        Simulator: A new Simulator object.
    """

    # load object
    simulate_method = 'default'
    if simulate_method == 'default':
        return Simulator(args)
    else:
        return NotImplementedError

##################################################


class Simulator:
    
    """
    Class for simulating phylogenetic datasets that can be converted into
    tensors with the Format step.
    """
    def __init__(self, args):
        """Initializes a new Simulator object.

        Args:
            args (dict): Contains phyddle settings.
        """
        
        # filesystem
        self.sim_prefix         = str(args['sim_prefix'])
        self.sim_dir            = str(args['sim_dir'])
        self.log_dir            = str(args['log_dir'])
        
        # resources
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = bool(args['use_parallel'])
        
        # simulate step config
        self.sim_command        = str(args['sim_command'])
        self.start_idx          = int(args['start_idx'])
        self.end_idx            = int(args['end_idx'])
        self.sim_more           = int(args['sim_more'])
        self.sim_batch_size     = int(args['sim_batch_size'])
        self.num_char           = int(args['num_char'])
        self.num_trees          = int(args['num_trees'])
        self.verbose            = bool(args['verbose'])
        
        # validate sim_command
        self.validate_sim_command()

        # set number of processors
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc

        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.rep_idx = []  # init with get_rep_idx

        # done
        return

    def get_rep_idx(self):
        """Determines replicate indices to use.

        This function will use the provided start and end index for new
        replicates, unless sim_more > 0. In that case, the function finds
        the largest replicate index, k, in sim_dir and then uses
        [k+1:k+sim_more] as replicate indices.

        Returns:
            int[]: List of replicate indices.

        """
        # if sim_more arg is defined, use it to overwrite rep_idx
        if self.sim_more > 0:
            rep_idx = set()
            files = os.listdir(f'{self.sim_dir}')
            for f in files:
                idx = None
                try:
                    idx = int(f.split('.')[1])
                except ValueError:
                    pass
                if idx is not None:
                    rep_idx.add(idx)

                #rep_idx.add(int(f.split('.')[1]))
            
            if len(list(rep_idx)) > 0:
                max_rep_idx = max(list(rep_idx))
                self.start_idx = max_rep_idx + 1
            else:
                self.start_idx = 0
            
            self.end_idx = self.start_idx + self.sim_more

        # determine rep_idx
        rep_idx = list(range(self.start_idx,
                             self.end_idx,
                             self.sim_batch_size))

        return rep_idx

    def validate_sim_command(self):
        """Reports error if sim_command is invalid.
        
        This function verifies that self.sim_command executes properly. It
        verifies the command and script exist. Then it runs a test job
        against a local temp directory, and verifies the correct output files
        with correct formats are generated. Validation then deletes any
        temporary files.
        """

        tok = self.sim_command.split(' ')
        
        if len(tok) < 2:
            msg = ( "Invalid sim_command setting. Command string "
                    f"'{self.sim_command}' is incomplete. A valid command "
                    "string is the command, then a space, then the relative "
                    "path to the simulation script: '[command] [sim_script]'.")
            util.print_err(msg)
            sys.exit()

        # get command and script
        cmd = tok[0]
        fn = tok[1]

        # verify executable exists
        if shutil.which(cmd) is None:
            msg = (f"Invalid sim_command setting. Command '{cmd}' not found.  "
                    "Please verify the current user can execute the command "
                    "from the local directory.")
            util.print_err(msg)
            sys.exit()

        # verify test script exists
        if not os.path.exists(fn):
            msg = (f"Invalid sim_command setting. Simulator script '{fn}' "
                    "not found. Please verify the script is visible to the "
                    "current user from the local directory.")
            util.print_err(msg)
            sys.exit()

        # todo:
        # run test job
        # check output
        # clean-up

        return

    def run(self):
        """Simulates training examples.

        This creates the target directory for new simulations then runs all
        simulation jobs.
        
        Simulation jobs are numbered by the replicate-index list (self.rep_idx).
        Each job is executed by calling self.sim_one(idx) where idx is a unique
        value in self.rep_idx.

        Each dispatched simulation task is expected to produce n=chunksize
        sequentially numbered new simulated datasets.
 
        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.
        """
        verbose = self.verbose

        # print header
        util.print_step_header('sim', None, self.sim_dir,
                               None, self.sim_prefix, verbose)

        # prepare workspace
        os.makedirs(self.sim_dir, exist_ok=True)
    
        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # simulate replicate IDs to generate
        self.rep_idx = self.get_rep_idx()
        
        # dispatch jobs
        util.print_str('▪ Simulating raw data', verbose)
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
                res = list(
                     tqdm(
                        pool.imap(self.sim_one, self.rep_idx, chunksize=1),
                        total=len(self.rep_idx),
                        desc='Simulating',
                        smoothing=0)
                     )
            
        else:
            # serial jobs
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx,
                                                      total=len(self.rep_idx),
                                                      desc='Simulating',
                                                      smoothing=0) ]

        # verify Simulate produced appropriate output for Format
        self.check_valid_output()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

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
        # tmp_fn     = f'{self.sim_dir}/{self.sim_prefix}.{idx}'
        cmd_str    = f'{self.sim_command} {self.sim_dir} {self.sim_prefix} {idx} {self.sim_batch_size}'
        # stdout_fn  = f'{tmp_fn}.stdout.log'
        # stderr_fn  = f'{tmp_fn}.stderr.log'
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
                # cmd_stdout = cmd_res.stdout.decode('UTF-8')
                # util.write_to_file(cmd_stdout, stdout_fn)
                # save stderr
                # cmd_stderr = cmd_res.stderr.decode('UTF-8')
                # if cmd_stderr != '':
                #    util.write_to_file(cmd_stderr, stderr_fn)
                # done simulating
                valid = True
            except subprocess.CalledProcessError:
                error_msg = 'generated a non-zero exit code.'
                self.logger.write_log('sim', f'Simulation {idx}: {error_msg}')
                num_attempt -= 1
                valid = False

        return
    
    def check_valid_output(self):
        """Checks that at least one sim_one call has valid output."""
        
        # get set of unique training example replicate indices
        sim_files = os.listdir(self.sim_dir)
        sim_prefix = [ x.split('.')[1] for x in sim_files ]
        sim_idx = set(sim_prefix)
        
        # collect all valid indices
        valid_phy = []
        valid_lbl = []
        valid_dat = []
        valid_all = []
        for idx in sim_idx:
            
            # check if replicate has tree, labels, and data files
            tmp_fn = f'{self.sim_dir}/{self.sim_prefix}.{idx}'
            has_phy = os.path.exists(f'{tmp_fn}.tre')
            if has_phy and self.num_trees > 0:
                valid_phy.append(idx)
                
            has_lbl = os.path.exists(f'{tmp_fn}.labels.csv')
            if has_lbl:
                valid_lbl.append(idx)
                
            has_dat = os.path.exists(f'{tmp_fn}.dat.csv') or \
                      os.path.exists(f'{tmp_fn}.dat.nex')
            
            if has_dat and self.num_char > 0:
                valid_dat.append(idx)
            
            # # replicate is valid if it has all three files
            # if has_phy and has_lbl and has_dat:
            #     valid_all.append(int(idx))

        # check files
        n_phy = len(valid_phy)
        n_lbl = len(valid_lbl)
        n_dat = len(valid_dat)
        num_rjust = len(str(max([n_phy,n_dat,n_lbl])))
        util.print_str(f'▪ Total counts of simulated files:', self.verbose)
        util.print_str(f'  ▪ ' + str(n_phy).rjust(num_rjust) + ' phylogeny files', self.verbose)
        util.print_str(f'  ▪ ' + str(n_dat).rjust(num_rjust) + ' data files', self.verbose)
        util.print_str(f'  ▪ ' + str(n_lbl).rjust(num_rjust) + ' labels files', self.verbose)
        
        # report any detected issues
        fail_phy = (len(valid_phy) == 0 and self.num_trees > 0)
        fail_dat = (len(valid_dat) == 0 and self.num_char > 0)
        if fail_phy or fail_dat:
            print('')
            if fail_phy:
                util.print_warn(f'{self.sim_dir} contains no phylogeny files, but num_tree > 0.')
            if fail_dat:
                util.print_warn(f'{self.sim_dir} contains no data files, but num_char > 0.')
            util.print_warn(f'Verify that simulation command:\n\n'
                            f'    {self.sim_command} {self.sim_dir} {self.sim_prefix} 0 1\n\n'
                            f'works as intended with the provided configuration.'
                            '\n')
            
        return
    
##################################################
