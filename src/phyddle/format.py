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
#from joblib import Parallel, delayed
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# phyddle imports
from phyddle import utilities

try:
    set_start_method('fork')
except RuntimeError:
    pass
#-----------------------------------------------------------------------------------------------------------------#

def load(args):

    # settings
    sys.setrecursionlimit(10000)

    # load object
    format_method = 'default'
    if format_method == 'default':
        return Formatter(args)
    else:
        return None

#-----------------------------------------------------------------------------------------------------------------#

class Formatter:

    def __init__(self, args): #, mdl):
        """
        Load the specified args and return the appropriate Formatter object.

        Args:
            args (dict): A dictionary containing the arguments.

        Returns:
            Formatter: The Formatter object based on the specified args.
        """
        self.set_args(args)
        self.logger = utilities.Logger(args)
        #self.model = mdl
        return        

    def set_args(self, args):
        """
        Set the arguments for the object.

        Args:
            args (dict): A dictionary of arguments.

        Returns:
            None
        """
        # formatter arguments
        self.args          = args
        self.verbose       = args['verbose']
        self.sim_dir       = args['sim_dir']
        self.fmt_dir       = args['fmt_dir']
        self.sim_proj      = args['sim_proj']
        self.fmt_proj      = args['fmt_proj']
        self.model_type    = args['model_type']
        self.model_variant = args['model_variant']
        self.tree_type     = args['tree_type']
        self.num_char      = args['num_char']
        self.num_states    = args['num_states']
        self.param_pred    = args['param_pred'] # parameters in label set (prediction)
        self.param_data    = args['param_data'] # parameters in data set (training, etc)
        self.chardata_format = args['chardata_format']
        self.tensor_format   = args['tensor_format']

        # encoder arguments
        self.model_name        = args['model_type']
        self.model_variant     = args['model_variant']
        self.tree_width_cats   = args['tree_width_cats']
        self.tree_encode_type  = args['tree_encode_type']
        self.char_encode_type  = args['char_encode_type']
        self.min_num_taxa      = args['min_num_taxa']
        self.max_num_taxa      = args['max_num_taxa']
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.use_parallel      = args['use_parallel']
        self.num_proc          = args['num_proc']
        # MJL: I think this was for breaking up large tensors into chunks
        self.tensor_part_size  = 500 # args['num_records_per_tensor_part']
        self.save_phyenc_csv   = args['save_phyenc_csv']
                    
        self.in_dir        = f'{self.sim_dir}/{self.sim_proj}'
        self.out_dir       = f'{self.fmt_dir}/{self.fmt_proj}'
        self.rep_idx       = list(range(self.start_idx, self.end_idx))

        self.num_tree_row = utilities.get_num_tree_row(self.tree_type, self.tree_encode_type)
        self.num_char_row = utilities.get_num_char_row(self.char_encode_type, self.num_char, self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row

        return

    def run(self):
        """
        Run the program.

        Returns:
            None
        """

        # print header
        utilities.print_step_header('fmt', [self.in_dir], self.out_dir, verbose=self.verbose)

        # new dir
        os.makedirs(self.out_dir, exist_ok=True)

        # build individual CDVS/CBLVS encodings
        utilities.print_str('▪ encoding raw data as tensors ...', verbose=self.verbose)
        self.encode_all()

        # actually fill and write full tensors
        utilities.print_str('▪ writing tensors ...', verbose=self.verbose)
        if self.tensor_format == 'csv':
            self.write_tensor_csv()
        elif self.tensor_format == 'hdf5':
            self.write_tensor_hdf5()

        utilities.print_str('... done!', verbose=self.verbose)

    
    def make_settings_str(self, idx, tree_width):
        """
        Create a settings string.

        Args:
            idx (int): The index.
            tree_width (int): The tree width.

        Returns:
            str: The settings string.
        """
        s = 'setting,value\n'
        s += 'sim_proj,'        + self.sim_proj + '\n'
        s += 'fmt_proj,'        + self.fmt_proj + '\n'
        s += 'model_type,'      + self.model_type + '\n'
        s += 'model_variant,'   + self.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'tree_width,'      + str(tree_width) + '\n'
        
        return s

    def encode_all(self):
        """
        Encode all replicates and return the result.
        
        If self.use_parallel is True, the encoding is done in parallel using multiple
        processes. Otherwise, the encoding is done sequentially.
        
        Returns:
            res (list): List of encoded replicates.
        """
        # visit each replicate, encode it, and return result
        if self.use_parallel:
            #res = Parallel(n_jobs=self.num_proc)(delayed(self.encode_one)(tmp_fn=f'{self.in_dir}/sim.{idx}', idx=idx) for idx in tqdm(self.rep_idx))
            args = [ (f'{self.in_dir}/sim.{idx}', idx) for idx in self.rep_idx ]
            with Pool(processes=self.num_proc) as pool:
                # res = pool.starmap(self.encode_one, tqdm(args,
                #            total=len(self.rep_idx),
                #            desc='Formatting'))
                
                res = list(tqdm(pool.imap(self.encode_one_star, args, chunksize=5),
                                total=len(args),
                                desc='Formatting'))
                # see https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm

                res = [ x for x in res ]
        else:
            res = [ self.encode_one(tmp_fn=f'{self.in_dir}/sim.{idx}', idx=idx) for idx in tqdm(self.rep_idx) ]

        # prepare phy_tensors
        self.phy_tensors = {}
        for size in self.tree_width_cats:
            self.phy_tensors[size] = {}

        # save all CBLVS/CDVS tensors into phy_tensors
        for i in range(len(res)):
            if res[i] is not None:
                tensor_size = res[i].shape[1]
                self.phy_tensors[tensor_size][i] = res[i]

        self.summ_stat_names = self.get_summ_stat_names()
        self.label_names = self.get_label_names()
        self.num_summ_stat = len(self.summ_stat_names)
        self.num_labels = len(self.label_names)
        return
    
    def get_summ_stat_names(self):
        """
        Get the names of the summary statistics from the first representative file.
    
        Returns:
            ret (list): List of summary statistic names.
        """
        # get first representative file
        idx = None
        for i in self.tree_width_cats:
            k_list = list( self.phy_tensors[i].keys() )
            if len(k_list) > 0 and idx is None:
                idx = k_list[0]
                
        #idx = list( self.phy_tensors[ self.tree_width_cats[0] ].keys() )[0]
        fn = f'{self.in_dir}/sim.{idx}.summ_stat.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret
    
    def get_label_names(self):
        """
        Get the names of the labels from the first representative file.
    
        Returns:
            ret (list): List of label names.
        """
        # get first representative file
        idx = None
        for i in self.tree_width_cats:
            k_list = list( self.phy_tensors[i].keys() )
            if len(k_list) > 0 and idx is None:
                idx = k_list[0]
        #idx = list( self.phy_tensors[ self.tree_width_cats[0] ].keys() )[0]
        fn = f'{self.in_dir}/sim.{idx}.param_row.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret


    def load_one_sim(self, idx, tree_width):
        """Load data for one simulation given its index and tree width.
    
        Args:
            idx (int): Index of the simulation.
            tree_width: Tree width for the simulation.
        
        Returns:
            Tuple of numpy arrays (x1, x2, x3).
        """
        #return None #(0,0,0)
    
        fname_base  = f'{self.in_dir}/sim.{idx}'
        fname_param = fname_base + '.param_row.csv'
        fname_stat  = fname_base + '.summ_stat.csv'
        x1 = self.phy_tensors[tree_width][idx].flatten()
        #dat_data[hdf5_idx,:] = x1 #phy_tensor.flatten()
        x2 = np.loadtxt(fname_stat, delimiter=',', skiprows=1)
        #dat_stat[hdf5_idx,:] = x2 #np.loadtxt(fname_stat, delimiter=',', skiprows=1)
        x3 = np.loadtxt(fname_param, delimiter=',', skiprows=1)
        #dat_labels[hdf5_idx,:] = x3 #np.loadtxt(fname_param, delimiter=',', skiprows=1)
        #x1,x2,x3=0,0,0
        return (x1,x2,x3)

    def write_tensor_hdf5(self):
        """
        Writes data to HDF5 file for each tree width.
        
        Parameters:
        - self: The instance of the class.
        
        Returns:
        - None
        """
        # get stat/label name info
        self.summ_stat_names_encode = [ s.encode('UTF-8') for s in self.summ_stat_names ]
        self.label_names_encode = [ s.encode('UTF-8') for s in self.label_names ]

        # build files
        for tree_width in sorted(list(self.phy_tensors.keys())):
                 
            # dimensions
            rep_idx = sorted(list(self.phy_tensors[tree_width]))
            num_samples = len(rep_idx)
            num_taxa = tree_width
            num_data_length = num_taxa * self.num_data_row
            #print(num_taxa, self.num_data_row)

            # print info
            print('Formatting {n} files for tree_type={tt} and tree_width={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_width))

            # HDF5 file
            out_hdf5_fn = f'{self.out_dir}/sim.nt{tree_width}.hdf5'
            hdf5_file = h5py.File(out_hdf5_fn, 'w')

            # name data
            dat_stat_names = hdf5_file.create_dataset('summ_stat_names', (1, self.num_summ_stat), 'S64', self.summ_stat_names_encode)
            dat_label_names = hdf5_file.create_dataset('label_names', (1, self.num_labels), 'S64', self.label_names_encode)

            # numerical data
            dat_data = hdf5_file.create_dataset('data', (num_samples, num_data_length), dtype='f', compression='gzip')
            dat_stat = hdf5_file.create_dataset('summ_stat', (num_samples, self.num_summ_stat), dtype='f', compression='gzip')
            dat_labels = hdf5_file.create_dataset('labels', (num_samples, self.num_labels), dtype='f', compression='gzip')

            # the replicates for this tree width
            _rep_idx = list(self.phy_tensors[tree_width].keys())
            
            # load all the info
            res = [ self.load_one_sim(idx=idx, tree_width=tree_width) for idx in tqdm(_rep_idx) ]
            
            # store all numerical data into hdf5
            dat_data[:,:] = np.vstack( [ x[0] for x in res ] )
            dat_stat[:,:] = np.vstack( [ x[1] for x in res ] )
            dat_labels[:,:] = np.vstack( [ x[2] for x in res ] )

            # read in summ_stats and labels (_all_ params) dataframes
            label_names_str = [ s.decode('UTF-8') for s in dat_label_names[0,:] ]
            summ_stat_names_str = [ s.decode('UTF-8') for s in dat_stat_names[0,:] ]
            df_summ_stats = pd.DataFrame( dat_stat, columns=summ_stat_names_str )
            df_labels = pd.DataFrame( dat_labels, columns=label_names_str )
            
            # separate data parameters (things we know) from label parameters (things we predict)
            df_labels_new = df_labels[self.param_pred]
            df_labels_move = df_labels[self.param_data]

            # concatenate new data parameters as column to existing summ_stats dataframe
            df_summ_stats_new = df_summ_stats.join( df_labels_move )

            # get new label/stat names
            new_label_names = self.param_pred
            new_summ_stat_names = self.summ_stat_names + self.param_data

            # delete original datasets 
            del hdf5_file['summ_stat']
            del hdf5_file['labels']
            del hdf5_file['label_names']
            del hdf5_file['summ_stat_names']
            
            # create new datasets
            hdf5_file.create_dataset('summ_stat', df_summ_stats_new.shape, 'f', df_summ_stats_new)
            hdf5_file.create_dataset('labels', df_labels_new.shape, 'f', df_labels_new)
            hdf5_file.create_dataset('label_names', (1, len(new_label_names)), 'S64', new_label_names)
            hdf5_file.create_dataset('summ_stat_names', (1, len(new_summ_stat_names)), 'S64', new_summ_stat_names)

            # close HDF5 files
            hdf5_file.close()

        return

    def process_one_param_hdf5(self, idx):
        """
        Process the parameters from an HDF5 file.
        
        Parameters:
        - self: The instance of the class.
        - idx: The index of the HDF5 file.
        
        Returns:
        - numpy array: The parameters loaded from the HDF5 file.
        """
        fname_base  = f'{self.in_dir}/sim.{idx}'
        fname_param = fname_base + '.param_row.csv'
        return np.loadtxt(fname_param, delimiter=',', skiprows=1)
        
    def process_one_stat_hdf5(self, idx):
        """
        Process the summary statistics from an HDF5 file.
        
        Parameters:
        - self: The instance of the class.
        - idx: The index of the HDF5 file.
        
        Returns:
        - numpy array: The summary statistics loaded from the HDF5 file.
        """
        fname_base  = f'{self.in_dir}/sim.{idx}'
        fname_stat  = fname_base + '.summ_stat.csv'
        return np.loadtxt(fname_stat, delimiter=',', skiprows=1)

    def process_one_label_hdf5(self, phy_tensor):
        """
        Process the labels from a phy_tensor.
        
        Parameters:
        - self: The instance of the class.
        - phy_tensor: The phy_tensor containing the labels.
        
        Returns:
        - numpy array: The labels flattened from the phy_tensor.
        """
        return phy_tensor.flatten() 
        
    def write_tensor_csv(self):
        """
        Writes CSV files for phylogenetic tensors.
        
        The method iterates through the phylogenetic tensors for each tree width and generates CSV files containing the tensor data and labels.
        
        Parameters:
            None
            
        Returns:
            None
        """
        # build files
        for tree_width in sorted(list(self.phy_tensors.keys())):
            
            # dimensions
            rep_idx = sorted(list(self.phy_tensors[tree_width]))
            num_samples = len(rep_idx)
            num_taxa = tree_width
            #num_data_length = num_taxa * self.num_data_row
            
            print('Formatting {n} files for tree_type={tt} and tree_width={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_width))
            
            # CSV files
            out_cblvs_fn  = f'{self.out_dir}/sim.nt{tree_width}.cblvs.data.csv'
            out_cdvs_fn   = f'{self.out_dir}/sim.nt{tree_width}.cdvs.data.csv'
            out_stat_fn   = f'{self.out_dir}/sim.nt{tree_width}.summ_stat.csv'
            out_labels_fn = f'{self.out_dir}/sim.nt{tree_width}.labels.csv'

            # cblvs tensor
            if self.tree_type == 'serial':
                with open(out_cblvs_fn, 'w') as outfile:
                    for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_width].items()):
                        fname = f'{self.in_dir}/sim.{idx}.cblvs.csv'
                        #with open(fname, 'r') as infile:
                        s = ','.join(str(a) for a in phy_tensor) + '\n' #infile.read()
                        z = outfile.write(s)
                    
                    # for j,i in enumerate(size_sort[tree_width]):
                    #     fname = self.in_dir + '/' + 'sim.' + str(i) + '.cblvs.csv'
                    #     with open(fname, 'r') as infile:
                    #         s = infile.read()
                    #         z = outfile.write(s)
                        

            # cdv file tensor       
            elif self.tree_type == 'extant':
                with open(out_cdvs_fn, 'w') as outfile:
                    #for j,i in enumerate(size_sort[tree_width]):
                    for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_width].items()):
                        fname = f'{self.in_dir}/sim.{idx}.cdvs.csv'
                        #with open(fname, 'r') as infile:
                        s = ','.join(map(str, phy_tensor.flatten())) + '\n'
                        #s = ','.join(str(a) for a in phy_tensor) + '\n' #infile.read()
                        #print(phy_tensor)
                        #print(s)
                        #xxxx
                        z = outfile.write(s)
                    
            # summary stats tensor
            with open(out_stat_fn, 'w') as outfile:
                for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_width].items()):
                #for j,i in enumerate(size_sort[tree_width]):
                    fname = f'{self.in_dir}/sim.{idx}.summ_stat.csv'
                    #fname = self.in_dir + '/' + 'sim.' + str(i) + '.summ_stat.csv'
                    with open(fname, 'r') as infile:
                        if j == 0:
                            s = infile.read()
                            z = outfile.write(s)
                        else:
                            s = ''.join(infile.readlines()[1:])
                            z = outfile.write(s)
                        

            # labels input tensor
            with open(out_labels_fn, 'w') as outfile:
                for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_width].items()):
                #for j,i in enumerate(size_sort[tree_width]):
                    #fname = self.in_dir + '/' + 'sim.' + str(i) + '.param_row.csv'
                    fname = f'{self.in_dir}/sim.{idx}.param_row.csv'
                    with open(fname, 'r') as infile:
                        if j == 0:
                            s = infile.read()
                            z = outfile.write(s)
                        else:
                            s = ''.join(infile.readlines()[1:])
                            z = outfile.write(s)

            
            # read in summ_stats and labels (_all_ params) dataframes
            df_summ_stats = pd.read_csv(out_stat_fn)  # original, contains _no_ parameters
            df_labels = pd.read_csv(out_labels_fn)    # original, contains _all_ parameters

            # separate data parameters (things we know) from label parameters (things we predict)
            df_labels_keep = df_labels[self.param_pred]
            df_labels_move = df_labels[self.param_data]

            # concatenate new data parameters as column to existing summ_stats dataframe
            df_summ_stats = df_summ_stats.join( df_labels_move )

            # overwrite original files with new modified versions
            df_summ_stats.to_csv(out_stat_fn, index=False)
            df_labels_keep.to_csv(out_labels_fn, index=False)

        return

    def encode_one_star(self, args):
        return self.encode_one(*args)

    def encode_one(self, tmp_fn, idx, save_phyenc_csv=False):

        """
        Generate a Google-style docstring for the given Python code.

        Parameters:
            None

        Returns:
            cpsv: numpy array
                Compact phylo-state tensor (CPST)

        Raises:
            None

        """
        NUM_DIGITS = 10
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')}, precision=NUM_DIGITS)
        
        # make filenames
        dat_nex_fn = tmp_fn + '.dat.nex'
        tre_fn     = tmp_fn + '.tre'
        prune_fn   = tmp_fn + '.extant.tre'
        cblvs_fn   = tmp_fn + '.cblvs.csv'
        cdvs_fn    = tmp_fn + '.cdvs.csv'
        ss_fn      = tmp_fn + '.summ_stat.csv'
        info_fn    = tmp_fn + '.info.csv'
        

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
        
        # state space
        #vecstr2int = self.model.states.vecstr2int #{ v:i for i,v in enumerate(int2vecstr) }


        # read in nexus data file
        if self.chardata_format == 'nexus':
            dat = utilities.convert_nexus_to_array(dat_nex_fn, self.char_encode_type, self.num_states)
        elif self.chardata_format == 'csv':
            dat = utilities.convert_csv_to_array(dat_nex_fn, self.char_encode_type, self.num_states)
        
        # get tree file
        phy = utilities.read_tree(tre_fn)
        if phy is None:
            return

        # prune tree, if needed
        if self.tree_type == 'extant':
            phy_prune = utilities.make_prune_phy(phy, prune_fn)
            if phy_prune is None:
                return # abort, no valid pruned tree
            else:
                phy = copy.deepcopy(phy_prune)  # valid pruned tree

        # get tree size
        num_taxa = len(phy.leaf_nodes())
        if num_taxa > np.max(self.tree_width_cats):
            return  # abort, too many taxa
        elif num_taxa < self.min_num_taxa or num_taxa < 0:
            return # abort, too few taxa

        # get tree width from resulting vector
        tree_width = utilities.find_tree_width(num_taxa, self.tree_width_cats)

        # create compact phylo-state tensor (CPST)
        cblvs = None
        cdvs = None

        # encode CBLVS
        if self.tree_type == 'serial':
            cblvs = self.encode_phy_tensor(phy, dat, tree_width=tree_width, tree_encode_type=self.tree_encode_type, tree_type='serial')
            cpsv = cblvs
            cpsv_fn = cblvs_fn

        # encode CDVS
        elif self.tree_type == 'extant':
            cdvs = self.encode_phy_tensor(phy, dat, tree_width=tree_width, tree_encode_type=self.tree_encode_type, tree_type='extant')
            cpsv = cdvs
            cpsv_fn = cdvs_fn

        # save CPSV
        save_phyenc_csv_ = self.save_phyenc_csv or save_phyenc_csv
        if save_phyenc_csv_ and cpsv is not None:
            cpsv_str = utilities.make_clean_phyloenc_str(cpsv.flatten())
            utilities.write_to_file(cpsv_str, cpsv_fn)

        # record info
        info_str = self.make_settings_str(idx, tree_width)
        utilities.write_to_file(info_str, info_fn)

        # record summ stat data
        ss     = self.make_summ_stat(phy, dat) #, vecstr2int)
        ss_str = self.make_summ_stat_str(ss)
        utilities.write_to_file(ss_str, ss_fn)
        
        # done!
        return cpsv
    

    # ==> move to Formatting? <==

    #def make_summ_stat(self, tre_fn, geo_fn, states_bits_str_inv):
    def make_summ_stat(self, phy, dat): #, states_bits_str_inv):
        """
        Generate summary statistics.

        Parameters:
        - phy: phylogenetic tree
        - dat: character data

        Returns:
        - summ_stats: dictionary containing summary statistics

        """
        # build summary stats
        summ_stats = {}

        # read tree + states
        #phy = dp.Tree.get(path=tre_fn, schema="newick")
        num_taxa                  = len(phy.leaf_nodes())
        #root_distances            = phy.calc_node_root_distances()
        #root_distances            = [ nd.root_distance for nd in phy.nodes() if nd.is_leaf]
        #phy.calc_node_ages(ultrametricity_precision=False)
        node_ages                 = phy.internal_node_ages(ultrametricity_precision=False)
        #tree_height               = np.max( root_distances )
        root_age                  = phy.seed_node.age
        branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]

        # tree statistics
        summ_stats['n_taxa']      = num_taxa
        summ_stats['tree_length'] = phy.length()
        summ_stats['root_age']    = root_age
        summ_stats['brlen_mean']  = np.mean(branch_lengths)
        summ_stats['brlen_var']   = np.var(branch_lengths)
        summ_stats['brlen_skew']  = sp.stats.skew(branch_lengths)
        #summ_stats['brlen_kurt']  = sp.stats.kurtosis(branch_lengths)
        summ_stats['age_mean']    = np.mean(node_ages)
        summ_stats['age_var']     = np.var(node_ages)
        summ_stats['age_skew']    = sp.stats.skew(node_ages)
        #summ_stats['age_kurt']    = sp.stats.kurtosis(root_distances)
        summ_stats['B1']          = dp.calculate.treemeasure.B1(phy)
        summ_stats['N_bar']       = dp.calculate.treemeasure.N_bar(phy)
        summ_stats['colless']     = dp.calculate.treemeasure.colless_tree_imbalance(phy)
        summ_stats['treeness']    = dp.calculate.treemeasure.treeness(phy)
        #summ_stats['gamma']       = dp.calculate.treemeasure.pybus_harvey_gamma(phy)
        #summ_stats['sackin']      = dp.calculate.treemeasure.sackin_index(phy)

        # read characters + states
        # f = open(geo_fn, 'r')
        # m = f.read().splitlines()
        # f.close()
        # y = re.search(string=m[2], pattern='NCHAR=([0-9]+)')
        # z = re.search(string=m[3], pattern='SYMBOLS="([0-9A-Za-z]+)"')
        # num_char = int(y.group(1))
        # states = z.group(1)
        # #num_states = len(states)
        # #num_combo = num_char * num_states

        # # get taxon data
        # taxon_state_block = m[ m.index('Matrix')+1 : m.index('END;')-1 ]
        # taxon_states = [ x.split(' ')[-1] for x in taxon_state_block ]

        # num_char = dat.shape[0]
        # taxon_states = []
        #print(dat)
        # for col in dat:
        #     taxon_states.append( ''.join([ str(x) for x in dat[col].to_list() ]) )

        # for col in range(dat.shape[1]):
        #     taxon_states.append( ''.join([ str(x) for x in dat.iloc[col].to_list() ]) )

        # freqs of entire char-set
        # freq_taxon_states = np.zeros(num_char, dtype='float')
        #print(dat)
        # get freqs of data-states, based on state encoding type
        if self.char_encode_type == 'integer':
            for i in range(self.num_states):
                summ_stats['f_dat_' + str(i)] = 0
            unique, counts = np.unique(dat, return_counts=True)
            for i,j in zip(unique, counts):
                summ_stats['f_dat_' + str(i)] = j / num_taxa
        
        elif self.char_encode_type == 'one_hot':
            for i in range(dat.shape[0]):
                #print(i)
                #print(np.sum(dat.iloc[i]))
                summ_stats['f_dat_' + str(i)] = np.sum(dat.iloc[i]) / num_taxa
        
        #summ_stats['f_char_' + str(i)] = 0.
        # for k in list(states_bits_str_inv.keys()):
        #     #freq_taxon_states[ states_bits_str_inv[k] ] = taxon_states.count(k) / num_taxa
        #     summ_stats['n_state_' + str(k)] = taxon_states.count(k)
        #     #summ_stats['f_state_' + str(k)] = taxon_states.count(k) / num_taxa
        #     for i,j in enumerate(k):
        #         if j != '0':
        #             summ_stats['n_char_' + str(i)] += summ_stats['n_state_' + k]
        #             #summ_stats['f_char_' + str(i)] += summ_stats['f_state_' + k]

        return summ_stats
    
    def make_summ_stat_str(self, ss):
        """
        Generate a string representation of the summary statistics.

        Parameters:
        - ss: dictionary containing summary statistics

        Returns:
        - keys_str: string containing the keys of the summary statistics
        - vals_str: string containing the values of the summary statistics

        """
        keys_str = ','.join( list(ss.keys()) ) + '\n'
        vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
        return keys_str + vals_str
    

    # ==> move to Formatting? <==
    def encode_phy_tensor(self, phy, dat, tree_width, tree_type, tree_encode_type, rescale=True):
        """
        Encode the phylogenetic tree and character data as a tensor.

        Parameters:
        - phy: phylogenetic tree
        - dat: character data
        - tree_width: width of the tree
        - tree_type: type of the tree ('serial' or 'extant')
        - tree_encode_type: type of tree encoding
        - rescale: boolean flag indicating whether to rescale the tensor

        Returns:
        - phy_tensor: encoded tensor

        Raises:
        - ValueError if tree_type is unrecognized

        """
        if tree_type == 'serial':
            phy_tensor = self.encode_cblvs(phy, dat, tree_width, tree_encode_type, rescale)
        elif tree_type == 'extant':
            phy_tensor = self.encode_cdvs(phy, dat, tree_width, tree_encode_type, rescale)
        else:
            ValueError(f'Unrecognized {tree_type}')
        return phy_tensor

    def encode_cdvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """Encode CDVs (Character-Dependent Values) based on a given phylogenetic tree.

        Args:
            phy (Phylo.Tree): The phylogenetic tree.
            dat (numpy.ndarray): The character data.
            tree_width (int): The width of the tree.
            tree_encode_type (str): The type of tree encoding.
            rescale (bool, optional): Whether to rescale the heights. Defaults to True.

        Returns:
            numpy.ndarray: The encoded CDVs tensor.

        Raises:
            ValueError: If an invalid tree_encode_type is provided.
        """
        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  internal node root-distance
        # 1:  leaf node branch length
        # 2:  internal node branch ength
        # 3+: state encoding
        
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

        # fill in phylo tensor
        # row 0: 
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor


    def encode_cblvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        
        """Encode CBLVs (Character-Based Length Values) based on a given phylogenetic tree.

        Args:
            phy (Phylo.Tree): The phylogenetic tree.
            dat (numpy.ndarray): The character data.
            tree_width (int): The width of the tree.
            tree_encode_type (str): The type of tree encoding.
            rescale (bool, optional): Whether to rescale the heights. Defaults to True.

        Returns:
            numpy.ndarray: The encoded CBLVs tensor.

        Raises:
            ValueError: If an invalid tree_encode_type is provided.
        """
        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  leaf node-to-last internal node distance
        # 1:  internal node root-distance
        # 2:  leaf node branch length
        # 3:  internal node branch ength
        # 4+: state encoding

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

        # fill in phylo tensor
        # 0: leaf brlen; 1: intnode brlen; 2:leaf-to-lastintnode len; 3:lastintnode-to-root len
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor
