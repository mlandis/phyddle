# standard packages
import os
import copy
#import re
#import csv

# external packages
import h5py
import pandas as pd
import numpy as np
import dendropy as dp
from joblib import Parallel, delayed
from tqdm import tqdm

# phyddle packages
import Utilities

class Formatter:

    def __init__(self, args, mdl):
        self.set_args(args)
        self.model = mdl
        return        

    def set_args(self, args):

        # formatter arguments
        self.args          = args
        self.proj          = args['proj']
        self.fmt_dir       = args['fmt_dir']
        self.sim_dir       = args['sim_dir']
        self.tree_type     = args['tree_type']
        self.num_char      = args['num_char']
        self.param_pred    = args['param_pred'] # parameters in label set (prediction)
        self.param_data    = args['param_data'] # parameters in data set (training, etc)
        self.tensor_format = args['tensor_format']

        # encoder arguments
        self.model_name        = args['model_type']
        self.model_variant     = args['model_variant']
        self.tree_sizes        = args['tree_sizes'] #[ 200, 500 ]
        self.min_num_taxa      = args['min_num_taxa']
        self.max_num_taxa      = args['max_num_taxa']
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.use_parallel      = args['use_parallel']
        self.num_proc          = args['num_proc']
        self.save_phyenc_csv   = args['save_phyenc_csv']

        if self.tree_type == 'serial':
            self.num_data_row = 4 + self.num_char
        if self.tree_type == 'extant':
            self.num_data_row = 3 + self.num_char
            
        self.in_dir        = f'{self.sim_dir}/{self.proj}'
        self.out_dir       = f'{self.fmt_dir}/{self.proj}'

        self.rep_idx       = list(range(self.start_idx, self.end_idx))

        return

    def run(self):

        # new dir
        os.makedirs(self.out_dir, exist_ok=True)

        # build individual CDVS/CBLVS encodings
        self.encode_all()

        # actually fill and write full tensors
        if self.tensor_format == 'csv':
            self.write_tensor_csv()
        elif self.tensor_format == 'hdf5':
            self.write_tensor_hdf5()
    

    def make_settings_str(self, idx, tree_width):

        s = 'setting,value\n'
        s += 'proj,'            + self.proj + '\n'
        s += 'model_type,'      + self.model.model_type + '\n'
        s += 'model_variant,'   + self.model.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'tree_width,'      + str(tree_width) + '\n'
        
        return s

    def encode_all(self):

        # visit each replicate, encode it, and return result
        if self.use_parallel:
            res = Parallel(n_jobs=self.num_proc)(delayed(self.encode_one)(tmp_fn=f'{self.in_dir}/sim.{idx}', idx=idx) for idx in tqdm(self.rep_idx))
        else:
            res = [ self.encode_one(tmp_fn=f'{self.in_dir}/sim.{idx}', idx=idx) for idx in tqdm(self.rep_idx) ]

        # prepare phy_tensors
        self.phy_tensors = {}
        for size in self.tree_sizes:
            self.phy_tensors[size] = {}

        # save all CBLVS/CDVS tensors into phy_tensors
        for i in range(len(res)):
            if res[i] is not None:
                tensor_size = res[i].shape[1]
                #tensor_length = len(res[i])
                #if self.tree_type == 'serial':
                #    tensor_size = tensor_length / (self.num_char + 4)
                #elif self.tree_type == 'extant':
                #    tensor_size = tensor_length /  (self.num_char + 3)
                #print(res[i])
                #print(tensor_length)
                #tensor_size = int(tensor_size)
                #print(i, tensor_size) #, tensor_length)
                self.phy_tensors[tensor_size][i] = res[i]

        self.summ_stat_names = self.get_summ_stat_names()
        self.label_names = self.get_label_names()
        self.num_summ_stat = len(self.summ_stat_names)
        self.num_labels = len(self.label_names)
        return
    
    def get_summ_stat_names(self):
        # get first representative file
        idx = list( self.phy_tensors[ self.tree_sizes[0] ].keys() )[0]
        fn = f'{self.in_dir}/sim.{idx}.summ_stat.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret
    
    def get_label_names(self):
        # get first representative file
        idx = list( self.phy_tensors[ self.tree_sizes[0] ].keys() )[0]
        fn = f'{self.in_dir}/sim.{idx}.param_row.csv'
        df = pd.read_csv(fn,header=0)
        ret = df.columns.to_list()
        return ret

    def write_tensor_hdf5(self):
        
        # get stat/label name info
        self.summ_stat_names_encode = [ s.encode('UTF-8') for s in self.summ_stat_names ]
        self.label_names_encode = [ s.encode('UTF-8') for s in self.label_names ]

        # build files
        for tree_size in sorted(list(self.phy_tensors.keys())):
                 
            # dimensions
            rep_idx = sorted(list(self.phy_tensors[tree_size]))
            num_samples = len(rep_idx)
            num_taxa = tree_size
            num_data_length = num_taxa * self.num_data_row

            # print info
            print('Formatting {n} files for tree_type={tt} and tree_size={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_size))

            # HDF5 file
            out_hdf5_fn = f'{self.out_dir}/sim.nt{tree_size}.hdf5'
            hdf5_file = h5py.File(out_hdf5_fn, 'w')

            # name data
            dat_stat_names = hdf5_file.create_dataset('summ_stat_names', (1, self.num_summ_stat), 'S64', self.summ_stat_names_encode)
            dat_label_names = hdf5_file.create_dataset('label_names', (1, self.num_labels), 'S64', self.label_names_encode)

            # numerical data
            dat_data = hdf5_file.create_dataset('data', (num_samples, num_data_length), dtype='f', compression='gzip')
            dat_stat = hdf5_file.create_dataset('summ_stat', (num_samples, self.num_summ_stat), dtype='f', compression='gzip')
            dat_labels = hdf5_file.create_dataset('labels', (num_samples, self.num_labels), dtype='f', compression='gzip')
      
            # store all numerical data into hdf5
            for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_size].items()):
                
                fname_base = f'{self.in_dir}/sim.{idx}'
                fname_param = fname_base + '.param_row.csv'
                fname_stat = fname_base + '.summ_stat.csv'

                dat_data[j,:] = phy_tensor.flatten()
                dat_stat[j,:] = np.loadtxt(fname_stat, delimiter=',', skiprows=1)
                dat_labels[j,:] = np.loadtxt(fname_param, delimiter=',', skiprows=1)

            #print(dat_data[j,:])
            
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

    def write_tensor_csv(self):
        # build files
        for tree_size in sorted(list(self.phy_tensors.keys())):
            
            # dimensions
            rep_idx = sorted(list(self.phy_tensors[tree_size]))
            num_samples = len(rep_idx)
            num_taxa = tree_size
            #num_data_length = num_taxa * self.num_data_row
            
            print('Formatting {n} files for tree_type={tt} and tree_size={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_size))
            
            # CSV files
            out_cblvs_fn  = f'{self.out_dir}/sim.nt{tree_size}.cblvs.data.csv'
            out_cdvs_fn   = f'{self.out_dir}/sim.nt{tree_size}.cdvs.data.csv'
            out_stat_fn   = f'{self.out_dir}/sim.nt{tree_size}.summ_stat.csv'
            out_labels_fn = f'{self.out_dir}/sim.nt{tree_size}.labels.csv'

            # cblvs tensor
            if self.tree_type == 'serial':
                with open(out_cblvs_fn, 'w') as outfile:
                    for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_size].items()):
                        fname = f'{self.in_dir}/sim.{idx}.cblvs.csv'
                        #with open(fname, 'r') as infile:
                        s = ','.join(str(a) for a in phy_tensor) + '\n' #infile.read()
                        z = outfile.write(s)
                    
                    # for j,i in enumerate(size_sort[tree_size]):
                    #     fname = self.in_dir + '/' + 'sim.' + str(i) + '.cblvs.csv'
                    #     with open(fname, 'r') as infile:
                    #         s = infile.read()
                    #         z = outfile.write(s)
                        

            # cdv file tensor       
            elif self.tree_type == 'extant':
                with open(out_cdvs_fn, 'w') as outfile:
                    #for j,i in enumerate(size_sort[tree_size]):
                    for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_size].items()):
                        fname = f'{self.in_dir}/sim.{idx}.cdvs.csv'
                        #with open(fname, 'r') as infile:
                        s = ','.join(str(a) for a in phy_tensor) + '\n' #infile.read()
                        z = outfile.write(s)
                    
            # summary stats tensor
            with open(out_stat_fn, 'w') as outfile:
                for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_size].items()):
                #for j,i in enumerate(size_sort[tree_size]):
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
                for j,(idx,phy_tensor) in enumerate(self.phy_tensors[tree_size].items()):
                #for j,i in enumerate(size_sort[tree_size]):
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

    def encode_one(self, tmp_fn, idx, save_phyenc_csv=False):

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
        
        # state space
        vecstr2int = self.model.states.vecstr2int #{ v:i for i,v in enumerate(int2vecstr) }

        # read in nexus data file
        dat = Utilities.convert_nexus_to_array(dat_nex_fn)

        # get tree file
        phy = Utilities.read_tree(tre_fn)
        if phy is None:
            return

        # prune tree, if needed
        if self.tree_type == 'extant':
            phy_prune = Utilities.make_prune_phy(phy, prune_fn)
            if phy_prune is None:
                return # abort, no valid pruned tree
            else:
                phy = copy.deepcopy(phy_prune)  # valid pruned tree

        # get tree size
        num_taxa = len(phy.leaf_nodes())
        if num_taxa > np.max(self.tree_sizes):
            return  # abort, too many taxa
        elif num_taxa < self.min_num_taxa or num_taxa < 0:
            return # abort, too few taxa

        # get tree width from resulting vector
        tree_width = Utilities.find_tree_width(num_taxa, self.tree_sizes)

        # create compact phylo-state tensor (CPST)
        cblvs = None
        cdvs = None

        # encode CBLVS
        if self.tree_type == 'serial':
            cblvs = self.encode_phy_tensor(phy, dat, tree_width=tree_width, tree_type='serial')
            cpsv = cblvs
            cpsv_fn = cblvs_fn

        # encode CDVS
        elif self.tree_type == 'extant':
            cdvs = self.encode_phy_tensor(phy, dat, tree_width=tree_width, tree_type='extant')
            cpsv = cdvs
            cpsv_fn = cdvs_fn

        # save CPSV
        save_phyenc_csv_ = self.save_phyenc_csv or save_phyenc_csv
        if save_phyenc_csv_ and cpsv is not None:
            cpsv_str = Utilities.make_clean_phyloenc_str(cpsv.flatten())
            Utilities.write_to_file(cpsv_str, cpsv_fn)

        # record info
        info_str = self.make_settings_str(idx, tree_width)
        Utilities.write_to_file(info_str, info_fn)

        # record summ stat data
        ss     = self.make_summ_stat(phy, dat, vecstr2int)
        ss_str = self.make_summ_stat_str(ss)
        Utilities.write_to_file(ss_str, ss_fn)
        
        # done!
        return cpsv
    

    # ==> move to Formatting? <==

    #def make_summ_stat(self, tre_fn, geo_fn, states_bits_str_inv):
    def make_summ_stat(self, phy, dat, states_bits_str_inv):

        # build summary stats
        summ_stats = {}

        # read tree + states
        #phy = dp.Tree.get(path=tre_fn, schema="newick")
        num_taxa                  = len(phy.leaf_nodes())
        root_distances            = phy.calc_node_root_distances()
        tree_height               = np.max( root_distances )
        branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]

        # tree statistics
        summ_stats['n_taxa']      = num_taxa
        summ_stats['tree_length'] = phy.length()
        summ_stats['tree_height'] = tree_height
        summ_stats['brlen_mean']  = np.mean(branch_lengths)
        summ_stats['brlen_var']   = np.var(branch_lengths)
        #summ_stats['brlen_skew']  = sp.stats.skew(branch_lengths)
        #summ_stats['brlen_kurt']  = sp.stats.kurtosis(branch_lengths)
        summ_stats['age_mean']    = np.mean(root_distances)
        summ_stats['age_var']     = np.var(root_distances)
        #summ_stats['age_skew']    = sp.stats.skew(root_distances)
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

        num_char = dat.shape[0]
        taxon_states = []
        #print(dat)
        for col in dat:
            taxon_states.append( ''.join([ str(x) for x in dat[col].to_list() ]) )

        # for col in range(dat.shape[1]):
        #     taxon_states.append( ''.join([ str(x) for x in dat.iloc[col].to_list() ]) )

        # freqs of entire char-set
        # freq_taxon_states = np.zeros(num_char, dtype='float')
        for i in range(num_char):
            summ_stats['n_char_' + str(i)] = 0
            #summ_stats['f_char_' + str(i)] = 0.
        for k in list(states_bits_str_inv.keys()):
            #freq_taxon_states[ states_bits_str_inv[k] ] = taxon_states.count(k) / num_taxa
            summ_stats['n_state_' + str(k)] = taxon_states.count(k)
            #summ_stats['f_state_' + str(k)] = taxon_states.count(k) / num_taxa
            for i,j in enumerate(k):
                if j != '0':
                    summ_stats['n_char_' + str(i)] += summ_stats['n_state_' + k]
                    #summ_stats['f_char_' + str(i)] += summ_stats['f_state_' + k]

        return summ_stats
    
    def make_summ_stat_str(self, ss):
        keys_str = ','.join( list(ss.keys()) ) + '\n'
        vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
        return keys_str + vals_str
    

    # ==> move to Formatting? <==
    def encode_phy_tensor(self, phy, dat, tree_width, tree_type, rescale=True):
        if tree_type == 'serial':
            phy_tensor = self.encode_cblvs(phy, dat, tree_width, rescale)
        elif tree_type == 'extant':
            phy_tensor = self.encode_cdvs(phy, dat, tree_width, rescale)
        else:
            ValueError(f'Unrecognized {tree_type}')
        return phy_tensor

    def encode_cdvs(self, phy, dat, tree_width, rescale=True):
        
        # num columns equals tree_size, 0-padding
        # returns tensor with following rows
        # 0: terminal brlen, 1: last-int-node brlen, 2: last-int-node root-dist
        
        # data dimensions
        num_char  = dat.shape[0]

        # initialize workspace
        root_distances = phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (3, tree_width) )
        states     = np.zeros( (num_char, tree_width) )
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
                heights[0,height_idx] = nd.edge.length
                states[:,state_idx]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                heights[1,height_idx] = nd.edge.length
                heights[2,height_idx] = nd.root_distance
                height_idx += 1

        # fill in phylo tensor
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor


    def encode_cblvs(self, phy, dat, tree_width, rescale=True):
        # data dimensions
        num_char   = dat.shape[0]

        # initialize workspace
        null       = phy.calc_node_root_distances(return_leaf_distances_only=False)
        heights    = np.zeros( (4, tree_width) ) 
        states     = np.zeros( (num_char, tree_width) )
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
                heights[0,height_idx] = nd.edge.length
                heights[2,height_idx] = nd.root_distance - last_int_node.root_distance
                states[:,state_idx]   = dat[nd.taxon.label].to_list()
                state_idx += 1
            else:
                #print(last_int_node.edge.length)
                heights[1,height_idx+1] = nd.edge.length
                heights[3,height_idx+1] = nd.root_distance
                last_int_node = nd
                height_idx += 1

        # fill in phylo tensor
        #heights.shape = (2, tree_size)
        # 0: leaf brlen; 1: intnode brlen; 2:leaf-to-lastintnode len; 3:lastintnode-to-root len
        if rescale:
            heights = heights / np.max(heights)
        phylo_tensor = np.vstack( [heights, states] )

        return phylo_tensor