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
import copy
import os
import sys

# external imports
import dendropy as dp
import h5py
import numpy as np
import scipy as sp
import pandas as pd
from multiprocessing import Pool, set_start_method, cpu_count
from tqdm import tqdm

# phyddle imports
from phyddle import utilities as util

# Uncomment to debug multiprocessing
# import multiprocessing.util as mp_util
# mp_util.log_to_stderr(mp_util.SUBDEBUG)

# Allows multiprocessing fork (not spawn) new processes on Unix-based OS.
# However, import phyddle.simulate also calls set_start_method('fork'), and the
# function throws RuntimeError when called the 2nd+ time within a single Python.
# We handle this with a try-except block.
try:
    set_start_method('fork')
except RuntimeError:
    pass

#------------------------------------------------------------------------------#

def load(args):
    """
    Load a Formatter object.

    This function creates an instance of the Formatter class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.
    """
    # settings
    sys.setrecursionlimit(10000)

    # load object
    format_method = 'default'
    if format_method == 'default':
        return Formatter(args)
    else:
        return NotImplementedError

#------------------------------------------------------------------------------#

class Formatter:
    """
    Class for formatting phylogenetic datasets and converting them into tensors
    to be used by the Train step.
    """
    def __init__(self, args): #, mdl):
        """
        Initializes a new Formatter object.

        Args:
            args (dict): Contains phyddle settings.
        """
        # initialize with phyddle settings
        self.set_args(args)
        # directory for simulations (input)
        self.sim_proj_dir = f'{self.sim_dir}/{self.sim_proj}'
        # directory for formatted tensors (output)
        self.fmt_proj_dir = f'{self.fmt_dir}/{self.fmt_proj}'
        # set number of processors
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc
        # run() attempts to generate one simulation per value in rep_idx,
        # where rep_idx is list of unique ints to identify simulated datasets
        if self.encode_all_sim:
            self.rep_idx = self.get_rep_idx()
        else:
            self.rep_idx = list(range(self.start_idx, self.end_idx))

        #  split_idx is later assigned a value, after raw data is encoded
        self.split_idx = {}

        # get size of CPV+S tensors
        self.num_tree_row = util.get_num_tree_row(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_row = util.get_num_char_row(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row

        # create logger to track runtime info
        self.logger = util.Logger(args)
        # done
        return        

    def set_args(self, args):
        """
        Assigns phyddle settings as Formatter attributes.

        Args:
            args (dict): Contains phyddle settings.

        """
        # formatter arguments
        self.args = args
        step_args = util.make_step_args('F', args)
        for k,v in step_args.items():
            setattr(self, k, v)

        # special case
        self.tree_width_cats = [ self.tree_width ] # will be removed

        return

    def run(self):
        """
        Formats all raw datasets into tensors.

        This method prints status updates, creates the target directory for new
        tensors, then formats each simulated raw dataset into an individual
        tensor-formatted dataset, then combines all individual tensors into
        a single tensor that contains all training examples.
        """
        
        verbose = self.verbose

        # print header
        util.print_step_header('fmt', [self.sim_proj_dir], self.fmt_proj_dir, verbose)
        
        # prepare workspace
        os.makedirs(self.fmt_proj_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # encode each dataset into individual tensors
        util.print_str('▪ Encoding raw data as tensors', verbose)
        self.encode_all()

        # write tensors across all examples to file
        util.print_str('▪ Combining and writing tensors', verbose)
        self.write_tensor()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        # util.print_str(f'▪ End time:     {end_time_str}', verbose)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)

    
    def make_settings_str(self, idx, tree_width):
        """
        Construct a string of settings for a single replicate.

        Args:
            idx (int): The replicate index.
            tree_width (int): The tree width.

        Returns:
            str: The settings string.
        """
        s =  'setting,value\n'
        s += f'sim_proj_dir,{self.sim_proj_dir}\n'
        s += f'fmt_proj_dir,{self.fmt_proj_dir}\n'
        s += f'replicate_index,{idx}\n'
        s += f'tree_width,{tree_width}\n'
        return s
    
    def encode_all(self):
        """
        Encode each simulated replicate into its own matrix format.
        
        Encode each simulated dataset identified by the replicate-index list
        (self.rep_idx). Each dataset is encoded by calling self.encode_one(idx)
        where idx is a unique value in self.rep_idx.

        Encoded simulations are then sorted by tree-width category and stored
        into the phy_tensors dictionary for later processing. Names and lengths
        of label and summary statistic lists are also saved.

        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.

        """

        # construct list of encoding arguments
        args = []

        for idx in self.rep_idx:
            args.append((f'{self.sim_proj_dir}/sim.{idx}', idx))

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

        # save all phylogenetic-state tensors into the phy_tensors dictionary,
        # while sorting tensors into different tree-width categories
        self.phy_tensors = {}
        # for size in self.tree_width_cats:
        #     self.phy_tensors[size] = {}
        for i in range(len(res)):
            if res[i] is not None:
                # tensor_size = res[i].shape[1]
                # self.phy_tensors[tensor_size][i] = res[i]
                self.phy_tensors[i] = res[i]

        # save names/lengths of summary statistic and label lists
        self.summ_stat_names = self.get_summ_stat_names()
        self.label_names     = self.get_label_names()
        self.num_summ_stat   = len(self.summ_stat_names)
        self.num_labels      = len(self.label_names)

        return
    
    def get_rep_idx(self):
        rep_idx = set()
        files = os.listdir(f'{self.sim_proj_dir}')
        for f in files:
            rep_idx.add(int(f.split('.')[1]))
        rep_idx = sorted(list(rep_idx))
        return rep_idx

    def get_summ_stat_names(self):
        """
        Get names of summary statistics from first representative file.
    
        Returns:
            ret (list): List of summary statistics names.
        """
        # get first representative file
        idx = None
        # for i in self.tree_width_cats:
        #     k_list = list(self.phy_tensors[i].keys())
        #     if len(k_list) > 0 and idx is None:
        #         idx = k_list[0]
        k_list = list(self.phy_tensors.keys())
        if len(k_list) > 0 and idx is None:
            idx = k_list[0]

        fn = f'{self.sim_proj_dir}/sim.{idx}.summ_stat.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret
    
    def get_label_names(self):
        """
        Get names of training labels from first representative file.
    
        Returns:
            ret (list): List of label names.
        """
        # get first representative file
        idx = None
        # for i in self.tree_width_cats:
        #     k_list = list(self.phy_tensors[i].keys())
        #     if len(k_list) > 0 and idx is None:
        #         idx = k_list[0]
        k_list = list(self.phy_tensors.keys())
        if len(k_list) > 0 and idx is None:
            idx = k_list[0]

        fn = f'{self.sim_proj_dir}/sim.{idx}.param_row.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret

    def split_examples(self):
        """
        Split examples into training and test datasets.
        """
        split_idx = {}
        # tree_width = self.tree_width_cats[0]
        # rep_idx = sorted(list(self.phy_tensors[tree_width]))
        rep_idx = sorted(list(self.phy_tensors.keys()))
        num_samples = len(rep_idx)
        np.random.shuffle(rep_idx)
        num_test = int(num_samples * self.prop_test)
        split_idx['test'] = rep_idx[:num_test]
        split_idx['train'] = rep_idx[num_test:]
            
        return split_idx

    def write_tensor(self):
        """
        Write the tensor in csv or hdf5 format.
        """
        self.split_idx = self.split_examples()
        if self.tensor_format == 'csv':
            # self.write_tensor_csv()
            self.write_tensor_csv('train')
            self.write_tensor_csv('test')
        elif self.tensor_format == 'hdf5':
            # self.write_tensor_hdf5()
            self.write_tensor_hdf5('train')
            self.write_tensor_hdf5('test')
        return


    def write_tensor_hdf5(self, data_str):
        """
        Writes data to HDF5 file for each tree width.

        This class creates HDF5 files

        """

        assert(data_str in ['test', 'train'])
        
        # build files
        tree_width = self.tree_width #self.tree_width_cats[0]
        #phy_tensor = self.phy_tensors[tree_width]
                 
        # dimensions
        rep_idx = self.split_idx[data_str]
        num_samples = len(rep_idx)
        num_data_length = tree_width * self.num_data_row

        # print info
        print(f'Making {data_str} hdf5 dataset: {num_samples} examples for tree width = {tree_width}')

        # HDF5 file
        out_hdf5_fn = f'{self.fmt_proj_dir}/{data_str}.nt{tree_width}.hdf5'
        hdf5_file = h5py.File(out_hdf5_fn, 'w')

        # create datasets for numerical data
        dat_data = hdf5_file.create_dataset('phy_data',
                                            (num_samples, num_data_length),
                                            dtype='f', compression='gzip')
        dat_stat = hdf5_file.create_dataset('summ_stat',
                                            (num_samples, self.num_summ_stat),
                                            dtype='f', compression='gzip')
        dat_labels = hdf5_file.create_dataset('labels',
                                                (num_samples, self.num_labels),
                                                dtype='f', compression='gzip')

        # the replicates for this tree width
        #_rep_idx = list(phy_tensor.keys())
        
        # load all the info
        res = [ self.load_one_sim(idx=idx, tree_width=tree_width) for idx in tqdm(rep_idx,
                            total=len(rep_idx),
                            desc='Combining',
                            smoothing=0) ]

        # store all numerical data into hdf5)
        if len(res) > 0:
            dat_data[:,:] = np.vstack( [ x[0] for x in res ] )
            dat_stat[:,:] = np.vstack( [ x[1] for x in res ] )
            dat_labels[:,:] = np.vstack( [ x[2] for x in res ] )

        # read in summ_stats and labels (_all_ params) dataframes
        df_summ_stats = pd.DataFrame(dat_stat, columns=self.summ_stat_names)
        df_labels = pd.DataFrame(dat_labels, columns=self.label_names)
        
        # separate data parameters (things we know) from label parameters (things we estimate)
        df_labels_new = df_labels[self.param_est]
        df_labels_move = df_labels[self.param_data]

        # concatenate new data parameters as column to existing summ_stats dataframe
        df_aux_data = df_summ_stats.join( df_labels_move )

        # get new label/stat names
        new_label_names = self.param_est
        new_aux_data_names = self.summ_stat_names + self.param_data

        # delete original datasets 
        del hdf5_file['summ_stat']
        del hdf5_file['labels']
        
        # create new datasets
        hdf5_file.create_dataset('labels', df_labels_new.shape, 'f', df_labels_new, compression='gzip')
        hdf5_file.create_dataset('label_names', (1, len(new_label_names)), 'S64', new_label_names, compression='gzip')
        hdf5_file.create_dataset('aux_data', df_aux_data.shape, 'f', df_aux_data, compression='gzip')
        hdf5_file.create_dataset('aux_data_names', (1, len(new_aux_data_names)), 'S64', new_aux_data_names, compression='gzip')

        # close HDF5 files
        hdf5_file.close()

        return
    
    def write_tensor_csv(self, data_str):
        """
        Writes CSV files for phylogenetic tensors.
        
        The method iterates through the phylogenetic tensors for each tree width
        and generates CSV files containing the tensor data and labels.
        """
        assert(data_str in ['test', 'train'])
        
        # build files
        tree_width = self.tree_width #_cats[0]
        phy_tensor = self.phy_tensors #[tree_width]

        # dimensions
        rep_idx = self.split_idx[data_str]
        num_samples = len(rep_idx)
            
        # info
        print(f'Making {data_str} csv dataset: {num_samples} examples for tree width = {tree_width}')
        
        # output csv filepaths
        #out_hdf5_fn = f'{self.fmt_proj_dir}/{data_str}.nt{tree_width}.hdf5'
        out_prefix    = f'{self.fmt_proj_dir}/{data_str}.nt{tree_width}'
        in_prefix     = f'{self.sim_proj_dir}/sim'
        out_phys_fn   = f'{out_prefix}.phy_data.csv'
        out_stat_fn   = f'{out_prefix}.aux_data.csv'
        out_labels_fn = f'{out_prefix}.labels.csv'

        # phylogenetic state tensor
        with open(out_phys_fn, 'w') as outfile:
            for idx in rep_idx:
                pt = phy_tensor[idx] 
            #for j,(idx,pt) in enumerate(phy_tensor.items()):
                s = ','.join(map(str, pt.flatten())) + '\n'
                outfile.write(s)

        # summary stats tensor
        with open(out_stat_fn, 'w') as outfile:
            is_first = True
            for idx in rep_idx:
            # for j,idx in enumerate(phy_tensor.keys()):
                # if idx in rep_idx:
                fname = f'{in_prefix}.{idx}.summ_stat.csv'
                with open(fname, 'r') as infile:
                    if is_first:
                        s = infile.read()
                        is_first = False
                    else:
                        s = ''.join(infile.readlines()[1:])
                    outfile.write(s)
                    
        # labels input tensor
        with open(out_labels_fn, 'w') as outfile:
            is_first = True
            for idx in rep_idx:
            # for j,idx in enumerate(phy_tensor.keys()):
                # if idx in rep_idx:
                fname = f'{in_prefix}.{idx}.param_row.csv'
                with open(fname, 'r') as infile:
                    if is_first:
                        s = infile.read()
                        is_first = False
                    else:
                        s = ''.join(infile.readlines()[1:])
                    outfile.write(s)

        # rearrange labels and summary statistics
        # - labels contains param_est
        # - aux_data contains summ_stat and param_data

        # read in summ_stats and labels
        df_summ_stats = pd.read_csv(out_stat_fn)
        df_labels = pd.read_csv(out_labels_fn)

        # separate data parameters (things we know) from label parameters (things we predict)
        df_labels_keep = df_labels[self.param_est]
        df_labels_move = df_labels[self.param_data]

        # concatenate new data parameters as column to existing summ_stats dataframe
        df_summ_stats = df_summ_stats.join( df_labels_move )

        # overwrite original files with new modified versions
        df_summ_stats.to_csv(out_stat_fn, index=False)
        df_labels_keep.to_csv(out_labels_fn, index=False)

        return
    
    def load_one_sim(self, idx, tree_width):
        """Load data for one simulation given its index and tree width.
    
        Args:
            idx (int): Index of the simulation.
            tree_width (int): Tree width for the simulation.
        
        Returns:
            Tuple of numpy arrays (x1, x2, x3).
        """
        fname_base  = f'{self.sim_proj_dir}/sim.{idx}'
        fname_param = fname_base + '.param_row.csv'
        fname_stat  = fname_base + '.summ_stat.csv'
        # x1 = self.phy_tensors[tree_width][idx].flatten()
        x1 = self.phy_tensors[idx].flatten()
        x2 = np.loadtxt(fname_stat, delimiter=',', skiprows=1)
        x3 = np.loadtxt(fname_param, delimiter=',', skiprows=1)
        return (x1,x2,x3)

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
        NUM_DIGITS = 10
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')},
                            precision=NUM_DIGITS)
        
        # make filenames
        dat_nex_fn = tmp_fn + '.dat.nex'
        tre_fn     = tmp_fn + '.tre'
        prune_fn   = tmp_fn + '.extant.tre'
        down_fn    = tmp_fn + '.downsampled.tre'
        cpsv_fn    = tmp_fn + '.phy_data.csv'
        ss_fn      = tmp_fn + '.summ_stat.csv'
        info_fn    = tmp_fn + '.info.csv'
        
        # check if key files exist
        err_msg = None
        if not os.path.exists(dat_nex_fn):
            err_msg = f'Formatter.encode_one(): {dat_nex_fn} does not exist'
            print(err_msg)
            self.logger.write_log('fmt', err_msg)
        if not os.path.exists(tre_fn):
            err_msg = f'Formatter.encode_one(): {tre_fn} does not exist'
            print(err_msg)
            self.logger.write_log('fmt', err_msg)
        if err_msg is not None:
            return
        
        # read in nexus data file as numpy array
        if self.char_format == 'nexus':
            dat = util.convert_nexus_to_array(dat_nex_fn,
                                                   self.char_encode,
                                                   self.num_states)
        elif self.char_format == 'csv':
            dat = util.convert_csv_to_array(dat_nex_fn,
                                                 self.char_encode,
                                                 self.num_states)
        
        # get tree file
        phy = util.read_tree(tre_fn)
        if phy is None:
            return

        # prune tree, if needed
        if self.tree_encode == 'extant':
            phy_prune = util.make_prune_phy(phy, prune_fn)
            if phy_prune is None:
                # abort, no valid pruned tree
                return
            else:
                # valid pruned tree
                phy = copy.deepcopy(phy_prune)

        # downsample taxa
        num_taxa_orig = len(phy.leaf_nodes())
        phy = util.make_downsample_phy(
            phy, down_fn,
            max_taxa=max(self.tree_width_cats),
            strategy=self.downsample_taxa)

        # get tree size
        num_taxa = len(phy.leaf_nodes())
        if num_taxa > np.max(self.tree_width_cats):
            # abort, too many taxa
            return
        if num_taxa < self.min_num_taxa or num_taxa < 0:
            # abort, too few taxa
            return

        # get tree width from resulting vector
        tree_width = util.find_tree_width(num_taxa, self.tree_width_cats)

        # create compact phylo-state vector, CPV+S = {CBLV+S, CDV+S}
        cpvs_data = None

        # encode CBLV+S
        cpvs_data = self.encode_cpvs(phy, dat, tree_width=tree_width,
                                     tree_encode_type=self.brlen_encode,
                                     tree_type=self.tree_encode)

        # save CPVS
        save_phyenc_csv_ = self.save_phyenc_csv or save_phyenc_csv
        if save_phyenc_csv_ and cpvs_data is not None:
            cpsv_str = util.make_clean_phyloenc_str(cpvs_data.flatten())
            util.write_to_file(cpsv_str, cpsv_fn)

        # record info
        info_str = self.make_settings_str(idx, tree_width)
        util.write_to_file(info_str, info_fn)

        # record summ stat data
        ss = self.make_summ_stat(phy, dat)
        # add downsampling info
        ss['num_taxa'] = num_taxa_orig
        ss['prop_taxa'] = num_taxa / num_taxa_orig

        # save summ. stats.
        ss_str = self.make_summ_stat_str(ss)
        util.write_to_file(ss_str, ss_fn)
        
        # done!
        return cpvs_data
    
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

        # read basic info from phylogenetic tree
        num_taxa                  = len(phy.leaf_nodes())
        node_ages                 = phy.internal_node_ages(ultrametricity_precision=False)
        root_age                  = phy.seed_node.age
        branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]
        #root_distances            = phy.calc_node_root_distances()
        #root_distances            = [ nd.root_distance for nd in phy.nodes() if nd.is_leaf]
        #phy.calc_node_ages(ultrametricity_precision=False)
        #tree_height               = np.max( root_distances )

        # tree statistics
        summ_stats['tree_length'] = phy.length()
        summ_stats['root_age']    = root_age
        summ_stats['brlen_mean']  = np.mean(branch_lengths)
        summ_stats['brlen_var']   = np.var(branch_lengths)
        summ_stats['brlen_skew']  = sp.stats.skew(branch_lengths)
        summ_stats['age_mean']    = np.mean(node_ages)
        summ_stats['age_var']     = np.var(node_ages)
        summ_stats['age_skew']    = sp.stats.skew(node_ages)
        summ_stats['B1']          = dp.calculate.treemeasure.B1(phy)
        summ_stats['N_bar']       = dp.calculate.treemeasure.N_bar(phy)
        summ_stats['colless']     = dp.calculate.treemeasure.colless_tree_imbalance(phy)
        summ_stats['treeness']    = dp.calculate.treemeasure.treeness(phy)

        # possible tree statistics, but not computable for arbitrary trees
        #summ_stats['gamma']       = dp.calculate.treemeasure.pybus_harvey_gamma(phy)
        #summ_stats['brlen_kurt']  = sp.stats.kurtosis(branch_lengths)
        #summ_stats['age_kurt']    = sp.stats.kurtosis(root_distances)
        #summ_stats['sackin']      = dp.calculate.treemeasure.sackin_index(phy)

        # frequencies of character states
        if self.char_encode == 'integer':
            # integer-encoded states
            for i in range(self.num_states):
                summ_stats['f_dat_' + str(i)] = 0
            unique, counts = np.unique(dat, return_counts=True)
            for i,j in zip(unique, counts):
                summ_stats['f_dat_' + str(i)] = j / num_taxa
        
        elif self.char_encode == 'one_hot':
            # one-hot-encoded states
            for i in range(dat.shape[0]):
                summ_stats['f_dat_' + str(i)] = np.sum(dat.iloc[i]) / num_taxa
        
        # done
        return summ_stats
    
    # ==> Can probably move to util? seems generic and useful
    def make_summ_stat_str(self, ss):
        """
        Generate a string representation of the summary statistics.

        Arguments:
            ss (dict): dictionary of summary statistics

        Returns:
            keys_str: string containing the keys of the summary statistics
            vals_str: string containing the values of the summary statistics

        """
        keys_str = ','.join( list(ss.keys()) ) + '\n'
        vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
        return keys_str + vals_str
    
    def encode_cpvs(self, phy, dat, tree_width, tree_type, tree_encode_type, rescale=True):
        """
        Encode Compact Phylogenetic Vector + States (CPV+S) array
        
        This function encodes the dataset into Compact Bijective Ladderized
        Vector + States (CBLV+S) when tree_type is 'serial' or Compact
        Diversity-Reordered Vector + States (CDV+S) when tree_type is 'extant'.

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa) in CPVS array
            tree_type (str):         type of the tree ('serial' or 'extant')
            tree_encode_type (str):  type of tree encoding ('height_only' or 'height_brlen')
            rescale:                 set tree height to 1 before encoding when True

        Returns:
            cpvs (numpy.array):      CPV+S encoded tensor
        """
        if tree_type == 'serial':
            cpvs = self.encode_cblvs(phy, dat, tree_width, tree_encode_type, rescale)
        elif tree_type == 'extant':
            cpvs = self.encode_cdvs(phy, dat, tree_width, tree_encode_type, rescale)
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
        # 2:  internal node branch ength
        # 3+: state encoding
        
        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa) in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or 'height_brlen')
            rescale:                 set tree height to 1 before encoding when True

        Returns:
            numpy.ndarray: The encoded CDVs tensor.
        """
        
        # data dimensions
        num_char_row = dat.shape[0]
        if tree_encode_type == 'height_only':
            num_tree_row = 1
        elif tree_encode_type == 'height_brlen':
            num_tree_row = 3

        # initialize workspace
        root_distances = phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (num_tree_row, tree_width) )
        states     = np.zeros( (num_char_row, tree_width) )
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
                heights[1,height_idx] = nd.edge.length
                states[:,state_idx]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                heights[0,height_idx] = nd.root_distance
                heights[2,height_idx] = nd.edge.length
                height_idx += 1

        # stack the phylo and states tensors
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor

    def encode_cblvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """
        Encode Compact Bijective Ladderized Vector + States (CBLV+S) array

        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  leaf node-to-last internal node distance
        # 1:  internal node root-distance
        # 2:  leaf node branch length
        # 3:  internal node branch ength
        # 4+: state encoding

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa) in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or 'height_brlen')
            rescale:                 set tree height to 1 before encoding when True

        Returns:
            numpy.ndarray: The encoded CDVs tensor.
        """

        # data dimensions
        num_char_row = dat.shape[0]
        if tree_encode_type == 'height_only':
            num_tree_row = 2
        elif tree_encode_type == 'height_brlen':
            num_tree_row = 4


        # initialize workspace
        null       = phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (num_tree_row, tree_width) ) 
        states     = np.zeros( (num_char_row, tree_width) )
        state_idx  = 0
        height_idx = 0

        # postorder traversal to rotate nodes by max-root-distance
        for nd in phy.postorder_node_iter():
            if nd.is_leaf():
                nd.max_root_distance = nd.root_distance
            else:
                children                  = nd.child_nodes()
                ch_max_root_distance      = [ ch.max_root_distance for ch in children ]
                ch_max_root_distance_rank = np.argsort( ch_max_root_distance )[::-1] # [0,1] or [1,0]
                children_reordered        = [ children[i] for i in ch_max_root_distance_rank ]
                nd.max_root_distance      = max(ch_max_root_distance)
                nd.set_children(children_reordered)

        # inorder traversal to fill matrix
        last_int_node = phy.seed_node
        last_int_node.edge.length = 0
        for nd in phy.inorder_node_iter():
            if nd.is_leaf():
                heights[0,height_idx] = nd.root_distance - last_int_node.root_distance
                heights[2,height_idx] = nd.edge.length
                states[:,state_idx]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                #print(last_int_node.edge.length)
                heights[1,height_idx+1] = nd.root_distance
                heights[3,height_idx+1] = nd.edge.length
                last_int_node = nd
                height_idx += 1

        # stack the phylo and states tensors
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor

#------------------------------------------------------------------------------#


    # def write_tensor_hdf5_old(self):
    #     """
    #     Writes data to HDF5 file for each tree width.

    #     This class creates HDF5 files

    #     """
        
    #     # build files
    #     for tree_width in sorted(list(self.phy_tensors.keys())):
                 
    #         # dimensions
    #         rep_idx = sorted(list(self.phy_tensors[tree_width]))
    #         num_samples = len(rep_idx)
    #         num_data_length = tree_width * self.num_data_row

    #         # print info
    #         print('Combining {n} files for tree_type={tt} and tree_width={ts}'.format(n=num_samples, tt=self.tree_encode, ts=tree_width))

    #         # HDF5 file
    #         out_hdf5_fn = f'{self.fmt_proj_dir}/sim.nt{tree_width}.hdf5'
    #         hdf5_file = h5py.File(out_hdf5_fn, 'w')

    #         # create datasets for numerical data
    #         dat_data = hdf5_file.create_dataset('phy_data',
    #                                             (num_samples, num_data_length),
    #                                             dtype='f', compression='gzip')
    #         dat_stat = hdf5_file.create_dataset('summ_stat',
    #                                             (num_samples, self.num_summ_stat),
    #                                             dtype='f', compression='gzip')
    #         dat_labels = hdf5_file.create_dataset('labels',
    #                                               (num_samples, self.num_labels),
    #                                               dtype='f', compression='gzip')

    #         # the replicates for this tree width
    #         _rep_idx = list(self.phy_tensors[tree_width].keys())
            
    #         # load all the info
    #         res = [ self.load_one_sim(idx=idx, tree_width=tree_width) for idx in tqdm(_rep_idx,
    #                             total=len(_rep_idx),
    #                             desc='Combining') ]

    #         # store all numerical data into hdf5)
    #         if len(res) > 0:
    #             dat_data[:,:] = np.vstack( [ x[0] for x in res ] )
    #             dat_stat[:,:] = np.vstack( [ x[1] for x in res ] )
    #             dat_labels[:,:] = np.vstack( [ x[2] for x in res ] )

    #         # read in summ_stats and labels (_all_ params) dataframes
    #         df_summ_stats = pd.DataFrame(dat_stat, columns=self.summ_stat_names)
    #         df_labels = pd.DataFrame(dat_labels, columns=self.label_names)
            
    #         # separate data parameters (things we know) from label parameters (things we estimate)
    #         df_labels_new = df_labels[self.param_est]
    #         df_labels_move = df_labels[self.param_data]

    #         # concatenate new data parameters as column to existing summ_stats dataframe
    #         df_aux_data = df_summ_stats.join( df_labels_move )

    #         # get new label/stat names
    #         new_label_names = self.param_est
    #         new_aux_data_names = self.summ_stat_names + self.param_data

    #         # delete original datasets 
    #         del hdf5_file['summ_stat']
    #         del hdf5_file['labels']
            
    #         # create new datasets
    #         hdf5_file.create_dataset('labels', df_labels_new.shape, 'f', df_labels_new, compression='gzip')
    #         hdf5_file.create_dataset('label_names', (1, len(new_label_names)), 'S64', new_label_names, compression='gzip')
    #         hdf5_file.create_dataset('aux_data', df_aux_data.shape, 'f', df_aux_data, compression='gzip')
    #         hdf5_file.create_dataset('aux_data_names', (1, len(new_aux_data_names)), 'S64', new_aux_data_names, compression='gzip')

    #         # close HDF5 files
    #         hdf5_file.close()

    #     return
         
    # def write_tensor_csv_old(self):
    #     """
    #     Writes CSV files for phylogenetic tensors.
        
    #     The method iterates through the phylogenetic tensors for each tree width
    #     and generates CSV files containing the tensor data and labels.
    #     """
    #     # build files
    #     for tree_width in sorted(list(self.phy_tensors.keys())):
            
    #         # helper variables
    #         phy_tensors = self.phy_tensors[tree_width]
    #         num_samples = len(phy_tensors)
            
    #         print('Formatting {n} files for tree_type={tt} and tree_width={ts}'.format(n=num_samples, tt=self.tree_encode, ts=tree_width))
            
    #         # output csv filepaths
    #         out_prefix    = f'{self.fmt_proj_dir}/sim.nt{tree_width}'
    #         in_prefix     = f'{self.sim_proj_dir}/sim'
    #         out_phys_fn   = f'{out_prefix}.phy_data.csv'
    #         out_stat_fn   = f'{out_prefix}.aux_data.csv'
    #         out_labels_fn = f'{out_prefix}.labels.csv'

    #         # phylogenetic state tensor
    #         with open(out_phys_fn, 'w') as outfile:
    #             for j,(idx,pt) in enumerate(phy_tensors.items()):
    #                 s = ','.join(map(str, pt.flatten())) + '\n'
    #                 outfile.write(s)

    #         # summary stats tensor
    #         with open(out_stat_fn, 'w') as outfile:
    #             for j,idx in enumerate(phy_tensors.keys()):
    #                 fname = f'{in_prefix}.{idx}.summ_stat.csv'
    #                 with open(fname, 'r') as infile:
    #                     if j == 0:
    #                         s = infile.read()
    #                     else:
    #                         s = ''.join(infile.readlines()[1:])
    #                     outfile.write(s)
                        
    #         # labels input tensor
    #         with open(out_labels_fn, 'w') as outfile:
    #             for j,idx in enumerate(phy_tensors.keys()):
    #                 fname = f'{in_prefix}.{idx}.param_row.csv'
    #                 with open(fname, 'r') as infile:
    #                     if j == 0:
    #                         s = infile.read()
    #                     else:
    #                         s = ''.join(infile.readlines()[1:])
    #                     outfile.write(s)

    #         # rearrange labels and summary statistics
    #         # - labels contains param_est
    #         # - aux_data contains summ_stat and param_data

    #         # read in summ_stats and labels
    #         df_summ_stats = pd.read_csv(out_stat_fn)
    #         df_labels = pd.read_csv(out_labels_fn)

    #         # separate data parameters (things we know) from label parameters (things we predict)
    #         df_labels_keep = df_labels[self.param_est]
    #         df_labels_move = df_labels[self.param_data]

    #         # concatenate new data parameters as column to existing summ_stats dataframe
    #         df_summ_stats = df_summ_stats.join( df_labels_move )

    #         # overwrite original files with new modified versions
    #         df_summ_stats.to_csv(out_stat_fn, index=False)
    #         df_labels_keep.to_csv(out_labels_fn, index=False)

    #     return