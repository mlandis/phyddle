#!/usr/bin/env python
"""
format
======
Defines classes and methods for the Format step. This step converts raw data
into tensor data that can be used by the Train step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""
# standard imports
# import copy
import os
import sys

# external imports
import dendropy as dp
import h5py
import numpy as np
# import scipy as sp
import pandas as pd
from multiprocessing import Pool, set_start_method, cpu_count
from tqdm import tqdm

# phyddle imports
from phyddle import utilities as util

# Uncomment to debug multiprocessing
# import multiprocessing.util as mp_util
# mp_util.log_to_stderr(mp_util.SUBDEBUG)

# Allows multiprocessing fork (not spawn) new processes on Unix-based OS.
# However, import phyddle.simulate also calls set_start_method('fork'), and
# the function throws RuntimeError when called the 2nd+ time within a single
# Python. We handle this with a try-except block.
try:
    set_start_method('fork')
except RuntimeError:
    pass

##################################################


def load(args):
    """Load a Formatter object.

    This function creates an instance of the Formatter class, initialized
    using phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.
    
    Returns:
        Simulator: A new Simulator object.
    """
    # settings
    sys.setrecursionlimit(10000)

    # load object
    format_method = 'default'
    if format_method == 'default':
        return Formatter(args)
    else:
        return NotImplementedError

##################################################


class Formatter:
    """
    Class for formatting phylogenetic datasets and converting them into
    tensors to be used by the Train step.
    """
    def __init__(self, args):
        """Initializes a new Formatter object.

        Args:
            args (dict): Contains phyddle settings.
        """
        
        # filesystem
        self.sim_prefix         = str(args['sim_prefix'])
        self.emp_prefix         = str(args['emp_prefix'])
        self.fmt_prefix         = str(args['fmt_prefix'])
        self.sim_dir            = str(args['sim_dir'])
        self.emp_dir            = str(args['emp_dir'])
        self.fmt_dir            = str(args['fmt_dir'])
        self.log_dir            = str(args['log_dir'])
        
        # analysis settings
        self.verbose            = bool(args['verbose'])
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = bool(args['use_parallel'])
        self.no_sim             = bool(args['no_sim'])
        self.no_emp             = bool(args['no_emp'])
        
        # dataset dimensions
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.min_num_taxa       = int(args['min_num_taxa'])
        self.max_num_taxa       = int(args['max_num_taxa'])
        self.tree_width         = int(args['tree_width'])
        
        # dataset processing
        self.encode_all_sim     = bool(args['encode_all_sim'])
        self.start_idx          = int(args['start_idx'])
        self.end_idx            = int(args['end_idx'])
        self.downsample_taxa    = str(args['downsample_taxa'])
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.char_format        = str(args['char_format'])
        self.tensor_format      = str(args['tensor_format'])
        self.param_est          = list(args['param_est'])
        self.param_data         = list(args['param_data'])
        self.prop_test          = float(args['prop_test'])
        self.log_offset         = float(args['log_offset'])
        self.save_phyenc_csv    = bool(args['save_phyenc_csv'])
        
        # set number of processors
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc

        # get size of CPV+S tensors
        num_tree_col = util.get_num_tree_col(self.tree_encode,
                                             self.brlen_encode)
        num_char_col = util.get_num_char_col(self.char_encode,
                                             self.num_char,
                                             self.num_states)
        self.num_data_col = num_tree_col + num_char_col
        
        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.rep_data           = dict()   # init with encode_all()
        self.split_idx          = dict()   # init with split_examples()
        self.rep_idx            = list()   # init with get_rep_idx()

        # done
        return

    def run(self):
        """Formats all raw datasets into tensors.

        This method prints status updates, creates the target directory for new
        tensors, then formats each simulated raw dataset into an individual
        tensor-formatted dataset, then combines all individual tensors into a
        single tensor that contains all training examples.
        """
        
        verbose = self.verbose

        valid_emp = self.has_valid_dataset(mode='emp')
        valid_sim = self.has_valid_dataset(mode='sim')
        input_print_dir = list()
        input_prefix = list()
        if valid_emp and not self.no_emp:
            input_print_dir.append(self.emp_dir)
            input_prefix.append(self.emp_prefix)
        if valid_sim and not self.no_sim:
            input_print_dir.append(self.sim_dir)
            input_prefix.append(self.sim_prefix)
            
        # print header
        util.print_step_header('fmt', input_print_dir, self.fmt_dir,
                               input_prefix, self.fmt_prefix, verbose)
        # prepare workspace
        os.makedirs(self.fmt_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        found_sim = False
        if self.no_sim:
            util.print_str('▪ Skipping simulated data', verbose)
        elif self.has_valid_dataset(mode='sim'):
            # run() attempts to generate one simulation per value in rep_idx,
            # where rep_idx is list of unique ints to identify simulated datasets
            util.print_str('▪ Collecting simulated data', verbose)
            self.rep_idx = self.get_rep_idx(mode='sim')
    
            # encode each dataset into individual tensors
            util.print_str('▪ Encoding simulated data as tensors', verbose)
            self.encode_all(mode='sim')
    
            # split examples into training and test datasets
            util.print_str('▪ Splitting into train and test datasets', verbose)
            self.split_idx = self.split_examples()
    
            # write tensors across all examples to file
            util.print_str('▪ Combining and writing simulated data as tensors', verbose)
            self.write_tensor(mode='sim')
            
            # done
            found_sim = True

        found_emp = False
        if self.no_emp:
            util.print_str('▪ Skipping empirical data', verbose)
        elif self.has_valid_dataset(mode='emp'):
            # collecting empirical files
            util.print_str('▪ Collecting empirical data', verbose)
            self.rep_idx = self.get_rep_idx(mode='emp')
        
            # encode each dataset into individual tensors
            util.print_str('▪ Encoding empirical data as tensors', verbose)
            self.encode_all(mode='emp')
    
            # write tensors across all examples to file
            util.print_str('▪ Combining and writing empirical data as tensors', verbose)
            self.split_idx = { 'empirical' : np.array(self.rep_idx) }
            self.write_tensor(mode='emp')
            
            # done
            found_emp = True

        # notify user if no work done
        if self.no_emp and self.no_sim:
            util.print_warn('Format has no work to do when no_sim '
                               'and no_emp are used together.')
        elif not found_sim and not found_emp:
            util.print_warn('No simulated or empirical datasets found. '
                            'Check config settings.', verbose)

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)
    
    def has_valid_dataset(self, mode='sim'):
        """Determines if empirical analysis is being performed.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
        
        Returns:
            bool: True if empirical analysis is being performed.
        """
        
        assert mode in ['sim', 'emp']
        dat_dir = ''
        dat_prefix = ''
        if mode == 'sim':
            dat_dir = self.sim_dir
            dat_prefix = self.sim_prefix
        elif mode == 'emp':
            dat_dir = self.emp_dir
            dat_prefix = self.emp_prefix
            
        # check if empirical directory exists
        if not os.path.exists(dat_dir):
            return False
        
        # check if empirical directory contains files
        files = set()
        for f in os.listdir(dat_dir):
            f = '.'.join( f.split('.')[0:2])
            if dat_prefix in f:
                files.add(f)
        
        if len(files) == 0:
            return False
        
        # check that at least one dataset is complete
        for f in files:
            has_dat = os.path.exists(f'{dat_dir}/{f}.dat.csv') or \
                      os.path.exists(f'{dat_dir}/{f}.dat.nex')
            has_tre = os.path.exists(f'{dat_dir}/{f}.tre')
            has_lbl = os.path.exists(f'{dat_dir}/{f}.labels.csv')
            
            # no labels needed for empirical datasets with no param_data
            if mode == 'emp' and len(self.param_data) == 0:
                has_lbl = True
            
            # if the 2-3 files exist, then we have 1+ valid datasets
            if has_dat and has_tre and has_lbl:
                return True
    
        return False

    def encode_all(self, mode='sim'):
        """Encode all simulated replicates.
        
        Encode each simulated dataset identified by the replicate-index list
        (self.rep_idx). Each dataset is encoded by calling self.encode_one(idx)
        where idx is a unique value in self.rep_idx.

        Encoded simulations are then sorted by tree-width category and stored
        into the phy_tensors dictionary for later processing. Names and lengths
        of label and summary statistic lists are also saved.

        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.

        Args:
            mode (str): specifies 'sim' or 'emp' dataset
        """

        # construct list of encoding arguments
        
        assert mode in ['sim', 'emp']
        encode_path = ''
        if mode == 'sim':
            encode_path = f'{self.sim_dir}/{self.sim_prefix}'
        elif mode == 'emp':
            encode_path = f'{self.emp_dir}/{self.emp_prefix}'
        
        args = []
        for idx in self.rep_idx:
            args.append((f'{encode_path}.{idx}', idx))
        
        # visit each replicate, encode it, and return result
        if self.use_parallel:
            # parallel jobs
            with Pool(processes=self.num_proc) as pool:
                # Note, it's critical to call this as list(tqdm(pool.imap(...)))
                # - pool.imap runs the parallelization
                # - tqdm generates progress bar
                # - list acts as a finalizer for the pool.imap work
                # Have not benchmarked imap/imap_unordered, chunksize, etc.
                res = list(tqdm(pool.imap(self.encode_one_star,
                                          args, chunksize=5),
                                total=len(args),
                                desc='Encoding',
                                smoothing=0))
                
        else:
            # serial jobs
            res = [ self.encode_one_star(a) for a in tqdm(args,
                                                          total=len(args),
                                                          desc='Encoding',
                                                          smoothing=0) ]

        num_total = len(res)
        num_valid = len([x for x in res if x is not None])
        util.print_str(f'Encoding found {num_valid} of {num_total} valid examples.')
        if num_valid == 0:
            # exits
            util.print_err('Format cannot proceed without valid examples. '
                           'Verify the simulation script generates '
                           'valid examples that are compatible with '
                           'the configuration (e.g. min_taxa_size setting).')
            sys.exit()
            
        # save all replicate output by index
        self.rep_data = {}
        for i in res:
            if i is not None:
                self.rep_data[i[0]] = { 'phy':i[1].flatten(),
                                        'aux': i[2],
                                        'lbl': i[3] }

        return
    
    def get_rep_idx(self, mode='sim'):
        """Determines replicate indices to use.

        This function finds all replicate indices within sim_proj_dir.

        Returns:
            int[]: List of replicate indices.
        """

        assert mode in ['sim', 'emp']
        
        # procedure assumes the simulate directory only contains the target set
        # of files which is an unsafe assumption, in general
        # find all rep index
        all_idx = []
        if self.encode_all_sim:
            all_idx = set()
            files = []
            if mode == 'sim':
                files = os.listdir(f'{self.sim_dir}')
            elif mode == 'emp':
                files = os.listdir(f'{self.emp_dir}')
            files = [ f for f in files if '.dat.' in f ]
            
            for f in files:
                s_idx = f.split('.')[1]
                try:
                    all_idx.add(int(s_idx))
                except ValueError:
                    print(f'Skipping invalid filename {f}.')
                    pass
                
            all_idx = sorted(list(all_idx))
        elif self.encode_all_sim:
            all_idx = list(range(self.start_idx, self.end_idx))
            
        return all_idx

    def split_examples(self):
        """Split examples into training and test datasets."""
        
        split_idx = {}
        rep_idx = sorted(list(self.rep_data.keys()))
        rep_idx = np.array(rep_idx)
        num_samples = len(rep_idx)
        
        # shuffle examples
        np.random.shuffle(rep_idx)
        
        # split examples
        num_test = int(num_samples * self.prop_test)
        split_idx['test'] = rep_idx[:num_test]
        split_idx['train'] = rep_idx[num_test:]
            
        # return split indices
        return split_idx

    def write_tensor(self, mode='train'):
        """Write tensors to file.
        
        This function writes the train and test tensors as files in csv
        or hdf5 format based on the tensor_format setting. Actual writing
        is delegated to write_tensor_csv() and write_tensor_hdf5() functions.
        """
        if mode == 'emp':
            if self.tensor_format == 'csv':
                self.write_tensor_csv('empirical')
            elif self.tensor_format == 'hdf5':
                self.write_tensor_hdf5('empirical')
        elif mode == 'sim':
            if self.tensor_format == 'csv':
                self.write_tensor_csv('train')
                self.write_tensor_csv('test')
            elif self.tensor_format == 'hdf5':
                self.write_tensor_hdf5('train')
                self.write_tensor_hdf5('test')
        return

    def write_tensor_hdf5(self, data_str):
        """Writes formatted tensors to HDF5 file.

        This function reads in all formatted input data the test or train
        datasets. Each input data point includes the phylogenetic state
        vector, the summary statistic vector, and the training label vector.
        Known parameters and summary statistics are converted into the
        auxiliary data vector. All data are then compiled into a tensor and
        stored into a new hdf5 object that is gzip-compressed, then written
        to file.

        Args:
            data_str (str): specifies 'test' or 'train' dataset
        
        """

        assert data_str in ['test', 'train', 'empirical']
        # mode = 'sim'
        # if data_str == 'empirical':
        #     mode = 'emp'
        
        # analysis info
        rep_idx               = self.split_idx[data_str]
        first_aux_data_values = list(self.rep_data.values())[0]['aux']
        first_par_est_values  = list(self.rep_data.values())[0]['lbl']
        aux_data_names        = first_aux_data_values.columns.to_list()
        par_est_names         = first_par_est_values.columns.to_list()
        
        # dimensions
        num_samples           = len(rep_idx)
        tree_width            = self.tree_width
        num_data_length       = tree_width * self.num_data_col
        num_aux_data          = len(aux_data_names)
        num_par_est           = len(par_est_names)
        
        # print info
        print(f'Making {data_str} hdf5 dataset: {num_samples} examples for tree width = {tree_width}')

        # HDF5 file
        out_hdf5_fn = f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}.hdf5'
        hdf5_file = h5py.File(out_hdf5_fn, 'w')

        # create HDF5 datasets
        hdf5_file.create_dataset('idx', rep_idx.shape,
                                 'i', rep_idx, compression='gzip' )
        hdf5_file.create_dataset('aux_data_names', (1, num_aux_data),
                                 'S64', aux_data_names, compression='gzip')
        hdf5_file.create_dataset('label_names',(1, num_par_est),
                                 'S64', par_est_names, compression='gzip')

        dat_phy = hdf5_file.create_dataset('phy_data',
                                           (num_samples, num_data_length),
                                           dtype='f', compression='gzip')
        dat_aux = hdf5_file.create_dataset('aux_data',
                                           (num_samples, num_aux_data),
                                           dtype='f', compression='gzip')
        dat_lbl = hdf5_file.create_dataset('labels',
                                           (num_samples, num_par_est),
                                           dtype='f', compression='gzip')

        # Each entry is a dictionary of phylo-state, aux. data, and label
        res = [ self.rep_data[idx] for idx in rep_idx ]
        
        # store all numerical data into hdf5)
        if len(res) > 0:
            dat_phy[:,:] = np.vstack( [ x['phy'] for x in res ] )
            dat_aux[:,:] = np.vstack( [ x['aux'] for x in res ] )
            dat_lbl[:,:] = np.vstack( [ x['lbl'] for x in res ] )

        # close HDF5 files
        hdf5_file.close()

        return
    
    def write_tensor_csv(self, data_str):
        """ Writes formatted tensor to CSV files.
        
        This function reads in all formatted input data the test or train
        datasets. Each input data point includes the phylogenetic state
        vector, the summary statistic vector, and the training label vector.
        Known parameters and summary statistics are converted into the
        auxiliary data vector. Data are then compiled into a different tensor
        types (phylo-state, aux. data, labels) and stored as separate CSV
        files.

        Args:
            data_str (str): specifies 'test' or 'train' dataset

        """
        assert data_str in ['test', 'train', 'empirical']
        
        # analysis info
        rep_idx               = self.split_idx[data_str]
        first_aux_data_values = list(self.rep_data.values())[0]['aux']
        first_par_est_values  = list(self.rep_data.values())[0]['lbl']
        aux_data_names        = first_aux_data_values.columns.to_list()
        par_est_names         = first_par_est_values.columns.to_list()

        # dimensions
        num_samples           = len(rep_idx)
        tree_width            = self.tree_width
        num_data_length       = tree_width * self.num_data_col
        num_aux_data          = len(aux_data_names)
        num_par_est           = len(par_est_names)
        
        # info
        print(f'Making {data_str} csv dataset: {num_samples} examples for tree width = {tree_width}')
        
        # output csv filepaths
        out_prefix    = f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}'
        in_prefix     = f'{self.sim_dir}/{self.sim_prefix}'
        out_idx_fn    = f'{out_prefix}.idx.csv'
        out_phy_fn    = f'{out_prefix}.phy_data.csv'
        out_aux_fn    = f'{out_prefix}.aux_data.csv'
        out_lbl_fn    = f'{out_prefix}.labels.csv'
        
        # phylo-state data tensor
        with open(out_phy_fn, 'w') as outfile:
            for idx in rep_idx:
                x = self.rep_data[idx]['phy']
                s = util.ndarray_to_flat_str(x) + '\n'
                outfile.write(s)

        # aux. data tensor
        with open(out_aux_fn, 'w') as outfile:
            is_first = True
            for idx in rep_idx:
                x = self.rep_data[idx]['aux']
                s = ''
                if is_first:
                    s += ','.join(aux_data_names) + '\n'
                s += util.ndarray_to_flat_str(x.to_numpy()) + '\n'
                outfile.write(s)
                    
        # labels tensor
        with open(out_lbl_fn, 'w') as outfile:
            is_first = True
            x = self.rep_data[idx]['lbl']
            s = ''
            if is_first:
                s += ','.join(par_est_names) + '\n'
            s += util.ndarray_to_flat_str(x.to_numpy()) + '\n'
            outfile.write(s)
            
        # replicate index
        df_idx = pd.DataFrame(rep_idx, columns=['idx'])
        df_idx.to_csv(out_idx_fn, index=False)

        return

    def encode_one_star(self, args):
        """Wrapper for encode_one w/ unpacked args"""
        return self.encode_one(*args)

    def encode_one(self, tmp_fn, idx, save_phyenc_csv=False):
        """
        Encode a single simulated raw dataset into tensor format.
         
        This function transforms raw input into tensor outputs. The inputs are
        read from a tree file and data matrix file. The tree is filtered to
        target ranges of taxon counts and pruned of non-extant taxa.
        
        Trees are then binned into tree-width categories (number of columns for
        compact vector representation). Next, trees and character data are
        encoded into extended CBLV or CDV formats that contain additional state
        and branch length information.

        Arguments:
            tmp_fn (str):            prefix for individual dataset
            idx (int):               replicate index for simulation
            save_phyenc_csv (bool):  save phylogenetic state tensor as csv?

        Returns:
            cpvs (numpy.array): compact phylo vector + states (CPVS)
        """
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')},
                            precision=util.OUTPUT_PRECISION)
        
        # make filenames
        dat_ext = ''
        if self.char_format == 'nexus':
            dat_ext = '.nex'
        elif self.char_format == 'csv':
            dat_ext = '.csv'
        
        dat_fn     = tmp_fn + '.dat' + dat_ext
        tre_fn     = tmp_fn + '.tre'
        prune_fn   = tmp_fn + '.extant.tre'
        down_fn    = tmp_fn + '.downsampled.tre'
        cpsv_fn    = tmp_fn + '.phy_data.csv'
        aux_fn     = tmp_fn + '.aux_data.csv'
        lbl_fn     = tmp_fn + '.labels.csv'
        par_est_fn = tmp_fn + '.param_est.csv'
        
        # check if key files exist
        if not os.path.exists(dat_fn):
            self.logger.write_log('fmt', f'Cannot find {dat_fn}')
            return
        if not os.path.exists(tre_fn):
            self.logger.write_log('fmt', f'Cannot find {tre_fn}')
            return
        
        # read in nexus data file as numpy array
        dat = None
        if self.char_format == 'nexus':
            dat = util.convert_nexus_to_array(dat_fn,
                                                   self.char_encode,
                                                   self.num_states)
            
        elif self.char_format == 'csv':
            dat = util.convert_csv_to_array(dat_fn,
                                                 self.char_encode,
                                                 self.num_states)
        
        # verify data dimensions match what phyddle expects
        expected_char_dim = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        
        if dat.shape[0] != expected_char_dim:
            # improve message with more details later
            msg = (f'The numbers of characters and states in the character data'
                   f' file {dat_fn} do not match the config file settings for'
                   f' num_char, num_states, and char_encode.')
            util.print_err(msg)
            # program quits
            
        # get tree file
        phy = util.read_tree(tre_fn)

        # abort if simulation is invalid
        if dat is None:
            self.logger.write_log('fmt', f'Cannot process {dat_fn}')
            return
        if phy is None:
            self.logger.write_log('fmt', f'Cannot process {tre_fn}')
            return

        # prune tree, if needed
        if self.tree_encode == 'extant':
            phy = util.make_prune_phy(phy, prune_fn)
            if phy is None:
                # abort, no valid pruned tree
                self.logger.write_log('fmt', f'Invalid pruned tree for {tre_fn}')
                return

        # downsample taxa
        num_taxa_orig = len(phy.leaf_nodes())
        phy = util.make_downsample_phy(
            phy, down_fn,
            max_taxa=self.tree_width,
            strategy=self.downsample_taxa)

        # get tree size
        num_taxa = len(phy.leaf_nodes())
        if num_taxa > self.tree_width:
            # abort, too many taxa
            self.logger.write_log('fmt', f'Too many taxa {tre_fn} has too many taxa')
            return
        if num_taxa < self.min_num_taxa or num_taxa < 0:
            # abort, too few taxa
            self.logger.write_log('fmt', f'Too few taxa (<{self.min_num_taxa}) for {tre_fn}')
            return

        # create compact phylo-state vector, CPV+S = {CBLV+S, CDV+S}
        cpvs_data = self.encode_cpvs(phy, dat, tree_width=self.tree_width,
                                     tree_encode_type=self.brlen_encode,
                                     tree_type=self.tree_encode, idx=idx)

        # save CPVS
        save_phyenc_csv_ = self.save_phyenc_csv or save_phyenc_csv
        if save_phyenc_csv_ and cpvs_data is not None:
            cpsv_str = util.ndarray_to_flat_str(cpvs_data.flatten()) + '\n'
            util.write_to_file(cpsv_str, cpsv_fn)

        # read in raw labels file
        labels = pd.read_csv(lbl_fn, header=0)
        
        # split raw labels into est vs. data
        param_est = labels[self.param_est]
        param_data = labels[self.param_data]
        
        # record summ stat data
        summ_stat = self.make_summ_stat(phy, dat)
        summ_stat['num_taxa'] = [ num_taxa_orig ]
        summ_stat['prop_taxa'] = [ num_taxa / num_taxa_orig ]
        
        # make aux. data from  "known" data parameters and sum. stats
        aux_data = pd.concat(objs=[summ_stat, param_data], axis=1)

        # save summ. stats.
        aux_data.to_csv(aux_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)
        
        # save param_est
        param_est.to_csv(par_est_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)

        # set empty param dataframes as None
        if len(param_est.columns) == 0:
            param_est = None

        # done!
        return idx, cpvs_data, aux_data, param_est
    
    def make_summ_stat(self, phy, dat):
        """
        Generate summary statistics.

        This function populates a dictionary of summary statistics. Keys
        are the names of statistics that later appear as column names in the
        summ_stat and aux_data files and containers. Values are for the
        inputted phy and dat variables.

        Arguments:
            phy (dendropy.Tree): phylogenetic tree
            dat (numpy.array): character data

        Returns:
            summ_stats (dict): summary statistics
        """
        # new dictionary to return
        summ_stats = {}

        # small dx to offset zero-valued things
        zero_offset = 1E-4
        one_offset = 1E+4
        
        # return default summ stats if phy not valid
        if phy is not None:
            
            # read basic info from phylogenetic tree
            num_taxa                  = len(phy.leaf_nodes())
            node_ages                 = phy.internal_node_ages(ultrametricity_precision=False)
            root_age                  = phy.seed_node.age
            branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]
            
            # tree statistics
            summ_stats['log10_tree_length'] = np.log10( phy.length() )
            summ_stats['log10_root_age']    = np.log10( root_age )
            summ_stats['log10_brlen_mean']  = np.log10( np.mean(branch_lengths) )
            summ_stats['log10_age_mean']    = np.log10( np.mean(node_ages) )
            summ_stats['log10_B1']          = np.log10( dp.calculate.treemeasure.B1(phy) )
            try:
                summ_stats['colless'] = np.log10( dp.calculate.treemeasure.colless_tree_imbalance(phy) )
            except ZeroDivisionError:
                summ_stats['colless'] = np.log10(zero_offset)
            summ_stats['age_var']     = np.log10( np.var(node_ages) )
            summ_stats['brlen_var']   = np.log10( np.var(branch_lengths) )
            summ_stats['treeness']    = np.log10( dp.calculate.treemeasure.treeness(phy) )
            summ_stats['N_bar']       = np.log10( dp.calculate.treemeasure.N_bar(phy) )

        # frequencies of character states
        if self.char_encode == 'integer':

            # initialize state freqs
            states = list(range(self.num_states))
            for i in states:
                summ_stats[f'f_dat_{i}'] = 0.0
                summ_stats[f'n_dat_{i}'] = 0.0

            # fill-in non-zero state freqs
            unique, counts = np.unique(dat, return_counts=True)
            for i,j in zip(states, counts):
                summ_stats[f'f_dat_{i}'] = j / num_taxa
                summ_stats[f'n_dat_{i}'] = j
                
            for i in states:
                f = summ_stats[f'f_dat_{i}']
                n = summ_stats[f'n_dat_{i}']
                if f == 0.0:
                    f = zero_offset
                elif f == 1.0:
                    f = one_offset
                else:
                    # print(f)
                    f = np.log(f / (1.0 - f))
                if n == 0.0:
                    n = zero_offset
                else:
                    n = np.log(n)
                summ_stats[f'f_dat_{i}'] = f
                summ_stats[f'n_dat_{i}'] = n
                
        elif self.char_encode == 'one_hot':
            # one-hot-encoded states
            for i in range(dat.shape[0]):
                f = np.sum(dat.iloc[i]) / num_taxa
                n = np.sum(dat.iloc[i])
                if f == 0.0:
                    f = zero_offset
                elif f == 1.0:
                    f = one_offset
                else:
                    f = np.log(f / (1.0 - f))
                if n == 0.0:
                    n = zero_offset
                else:
                    n = np.log(n)
                summ_stats['f_dat_' + str(i)] = f
                summ_stats['n_dat_' + str(i)] = n

        df = pd.DataFrame(summ_stats, index=[0])

        # done
        return df
    
    def encode_cpvs(self, phy, dat, tree_width, tree_type,
                    tree_encode_type, idx, rescale=True):
        """
        Encode Compact Phylogenetic Vector + States (CPV+S) array
        
        This function encodes the dataset into Compact Bijective Ladderized
        Vector + States (CBLV+S) when tree_type is 'serial' or Compact
        Diversity-Reordered Vector + States (CDV+S) when tree_type is 'extant'.

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_type (str):         type of the tree ('serial' or 'extant')
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            idx (int):               replicate index
            rescale (bool):          set tree height to 1 then encode, if True

        Returns:
            cpvs (numpy.array):      CPV+S encoded tensor
        """
        # taxon labels must match for each phy and dat replicate
        phy_labels = set([ n.taxon.label for n in phy.leaf_nodes() ])
        dat_labels = set(dat.columns.to_list())
        phy_missing = phy_labels.difference(dat_labels)
        if len(phy_missing) != 0:
            phy_missing = sorted(list(phy_missing))
            # dat_missing = sorted(list(set(dat_labels).difference(set(phy_labels))))
            err_msg = f'Missing taxon labels in dat but not in phy for replicate {idx}: '
            # if len(phy_missing) > 0:
            err_msg += ' '.join(phy_missing)
            # if len(dat_missing) > 0:
            #    err_msg += f' Missing from dat: {' '.join(dat_missing)}.'
            raise ValueError(err_msg)
        
        cpvs = None
        if tree_type == 'serial':
            cpvs = self.encode_cblvs(phy, dat, tree_width,
                                     tree_encode_type, rescale)
        elif tree_type == 'extant':
            cpvs = self.encode_cdvs(phy, dat, tree_width,
                                    tree_encode_type, rescale)
        else:
            ValueError(f'Unrecognized {tree_type}')
        return cpvs

    def encode_cdvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """
        Encode Compact Diversity-reordered Vector + States (CDV+S) array

        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  internal node root-distance
        # 1:  leaf node branch length
        # 2:  internal node branch length
        # 3+: state encoding
        
        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            rescale:                 set tree height to 1 then encode, if True

        Returns:
            numpy.ndarray: The encoded CDV+S tensor.
        """
        
        # data dimensions
        num_tree_col = 0
        num_char_col = dat.shape[0]
        if tree_encode_type == 'height_only':
            num_tree_col = 1
        elif tree_encode_type == 'height_brlen':
            num_tree_col = 3

        # initialize workspace
        phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (tree_width, num_tree_col) )
        states     = np.zeros( (tree_width, num_char_col) )
        state_idx  = 0
        height_idx = 0

        # postorder traversal to rotate nodes by clade-length
        for nd in phy.postorder_node_iter():
            if nd.is_leaf():
                nd.treelen = 0.
            else:
                children           = nd.child_nodes()
                ch_treelen         = [ (ch.edge.length + ch.treelen) for ch in children ]
                nd.treelen         = sum(ch_treelen)
                ch_treelen_rank    = np.argsort( ch_treelen )[::-1]
                children_reordered = [ children[i] for i in ch_treelen_rank ]
                nd.set_children(children_reordered)

        # inorder traversal to fill matrix
        phy.seed_node.edge.length = 0
        for nd in phy.inorder_node_iter():
            
            if nd.is_leaf():
                heights[height_idx,1] = nd.edge.length
                states[state_idx,:]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                heights[height_idx,0] = nd.root_distance
                heights[height_idx,2] = nd.edge.length
                height_idx += 1

        # stack the phylo and states tensors
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.hstack( [heights, states] )

        return phylo_tensor

    def encode_cblvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """
        Encode Compact Bijective Ladderized Vector + States (CBLV+S) array

        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  leaf node-to-last internal node distance
        # 1:  internal node root-distance
        # 2:  leaf node branch length
        # 3:  internal node branch length
        # 4+: state encoding

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            rescale:                 set tree height to 1 then encode, if True

        Returns:
            numpy.ndarray: The encoded CBLV+S tensor.
        """

        # data dimensions
        num_tree_col = 0
        num_char_col = dat.shape[0]
        if tree_encode_type == 'height_only':
            num_tree_col = 2
        elif tree_encode_type == 'height_brlen':
            num_tree_col = 4

        # initialize workspace
        phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (tree_width, num_tree_col) )
        states     = np.zeros( (tree_width, num_char_col) )
        state_idx  = 0
        height_idx = 0

        # postorder traversal to rotate nodes by max-root-distance
        for nd in phy.postorder_node_iter():
            if nd.is_leaf():
                nd.max_root_distance = nd.root_distance
            else:
                children                  = nd.child_nodes()
                ch_max_root_distance      = [ ch.max_root_distance for ch in children ]
                ch_max_root_distance_rank = np.argsort( ch_max_root_distance )[::-1]  # [0,1] or [1,0]
                children_reordered        = [ children[i] for i in ch_max_root_distance_rank ]
                nd.max_root_distance      = max(ch_max_root_distance)
                nd.set_children(children_reordered)

        # inorder traversal to fill matrix
        last_int_node = phy.seed_node
        last_int_node.edge.length = 0
        for nd in phy.inorder_node_iter():
            if nd.is_leaf():
                heights[height_idx,0] = nd.root_distance - last_int_node.root_distance
                heights[height_idx,2] = nd.edge.length
                states[state_idx,:]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                heights[height_idx+1,1] = nd.root_distance
                heights[height_idx+1,3] = nd.edge.length
                last_int_node = nd
                height_idx += 1

        # stack the phylo and states tensors
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.hstack( [heights, states] )

        return phylo_tensor

##################################################
