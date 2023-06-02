import os
import csv
import h5py
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


import Utilities

class Formatter:

    def __init__(self, args, mdl):
        self.set_args(args)
        self.model = mdl
        return        

    def set_args(self, args):

        # formatter arguments
        self.args          = args
        self.job_name      = args['job_name']
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
        self.tree_sizes        = [ 200, 500 ]
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.use_parallel      = args['use_parallel']
        self.num_proc          = args['num_proc']

        if self.tree_type == 'serial':
            self.num_data_row = 2 + self.num_char
        if self.tree_type == 'extant':
            self.num_data_row = 1 + self.num_char
            
        self.in_dir        = f'{self.sim_dir}/{self.job_name}'
        self.out_dir       = f'{self.fmt_dir}/{self.job_name}'

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
    

    def make_settings_str(self, idx, mtx_size):

        s = 'setting,value\n'
        s += 'job_name,' + self.job_name + '\n'
        s += 'model_type,' + self.model.model_type + '\n'
        s += 'model_variant,' + self.model.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'taxon_category,' + str(mtx_size) + '\n'
        
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
                tensor_length = len(res[i])
                if self.tree_type == 'serial':
                    tensor_size = tensor_length / (self.num_char + 2)
                elif self.tree_type == 'extant':
                    tensor_size = tensor_length /  (self.num_char + 1)
                print(res[i])
                print(tensor_length)
                tensor_size = int(tensor_size)
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

                dat_data[j,:] = phy_tensor
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
                        with open(fname, 'r') as infile:
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
                        with open(fname, 'r') as infile:
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

    def encode_one(self, tmp_fn, idx):

        NUM_DIGITS = 10
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')}, precision=NUM_DIGITS)
        
        # make filenames
        geo_fn    = tmp_fn + '.geosse.nex'
        tre_fn    = tmp_fn + '.tre'
        prune_fn  = tmp_fn + '.extant.tre'
        nex_fn    = tmp_fn + '.nex'
        cblvs_fn  = tmp_fn + '.cblvs.csv'
        cdvs_fn   = tmp_fn + '.cdvs.csv'
        ss_fn     = tmp_fn + '.summ_stat.csv'
        info_fn   = tmp_fn + '.info.csv'
        
        # state space
        int2vec    = self.model.states.int2vec
        int2vecstr = self.model.states.int2vecstr #[ ''.join([str(y) for y in x]) for x in int2vec ]
        vecstr2int = self.model.states.vecstr2int #{ v:i for i,v in enumerate(int2vecstr) }

        # verify tree size & existence!
        #result_str     = ''
        n_taxa_idx     = Utilities.get_num_taxa(tre_fn) #, idx, self.tree_sizes)
        taxon_size_idx = Utilities.find_taxon_size(n_taxa_idx, self.tree_sizes)

        # handle simulation based on tree size
        if n_taxa_idx > np.max(self.tree_sizes):
            # too many taxa
            return
        elif n_taxa_idx <= 0:
            # too few taxa
            return
        else:
            # valid number of taxa
            # generate extinct-pruned tree
            prune_success = Utilities.make_prune_phy(tre_fn, prune_fn)

            # generate nexus file 0/1 ranges
            taxon_states,nexus_str = Utilities.convert_nex(nex_fn, tre_fn, int2vec)
            Utilities.write_to_file(nexus_str, geo_fn)

            # then get CBLVS working
            cblv,new_order = Utilities.vectorize_tree(tre_fn, max_taxa=taxon_size_idx, prob=1.0 )
            cblvs = Utilities.make_cblvs_geosse(cblv, taxon_states, new_order)
            print('cblvs', cblvs.shape)
        
            # NOTE: this if statement should not be needed, but for some reason the "next"
            # seems to run even when make_prune_phy returns False
            # generate CDVS file
            if prune_success:
                cdvs = Utilities.make_cdvs(prune_fn, taxon_size_idx, taxon_states, int2vecstr)
                print('cdvs:', cdvs.shape)
            
            # output files
            mtx_size = cblv.shape[1]


        # record info
        info_str = self.make_settings_str(idx, mtx_size)
        Utilities.write_to_file(info_str, info_fn)

        # record CBLVS data
        cblvs_str = np.array2string(cblvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
        cblvs_str = cblvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
        #cblvs_str = Utilities.clean_scientific_notation(cblvs_str)
        Utilities.write_to_file(cblvs_str, cblvs_fn)

        # record CDVS data
        if prune_success:
            cdvs_str = np.array2string(cdvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
            cdvs_str = cdvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
            #cdvs_str = Utilities.clean_scientific_notation(cdvs_str)
            Utilities.write_to_file(cdvs_str, cdvs_fn)

        # record summ stat data
        ss = Utilities.make_summ_stat(tre_fn, geo_fn, vecstr2int)
        ss_str = Utilities.make_summ_stat_str(ss)
        #ss_str = Utilities.clean_scientific_notation(ss_str) #re.sub( '\.0+E\+0+', '', ss_str)
        Utilities.write_to_file(ss_str, ss_fn)

        if self.tree_type == 'extant':
            data = cdvs
        elif self.tree_type == 'serial':
            data = cblvs

        return data
    

    # def make_tensors_old(self):
    #         os.makedirs(self.out_dir, exist_ok=True)

    #         # collect files with replicate info
    #         files = os.listdir(self.in_dir)
    #         info_files = [ x for x in files if 'info' in x ]

    #         # sort replicate indices into size-category lists
    #         first_valid_file = True
    #         size_sort = {}
    #         for fn in info_files:
    #             fn = self.sim_dir + '/' + self.job_name + '/' + fn
    #             idx = -1
    #             size = -1
    #             all_files_valid = False

    #             with open(fn, newline='') as csvfile:
    #                 info = csv.reader(csvfile, delimiter=',')
    #                 for row in info:
    #                     if row[0] == 'replicate_index':
    #                         idx = int(row[1])
    #                     elif row[0] == 'taxon_category':
    #                         size = int(row[1])
                        
    #                 # check that all necessary files exist
    #                 #all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cdvs.csv','cblvs.csv','param2.csv','summ_stat.csv']]
    #                 if self.tree_type == 'serial':
    #                     all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cblvs.csv','param_row.csv','summ_stat.csv']]
    #                 elif self.tree_type == 'extant':
    #                     all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cdvs.csv','param_row.csv','summ_stat.csv']]
    #                 else:
    #                     raise NotImplementedError
    #                 all_files_valid = all( [os.path.isfile(fn) for fn in all_files] )

    #                 # place index into tree_size category if all necessary files exist
    #                 if all_files_valid:
    #                     if size >= 0 and size not in size_sort:
    #                         size_sort[size] = []
    #                     if size >= 0 and idx >= 0:
    #                         size_sort[size].append(idx)

    #                     # collect simpler header info from summ_stat and param_row 
    #                     if first_valid_file:
    #                         prefix = f'{self.in_dir}/sim.{idx}'
    #                         summ_stat = pd.read_csv(prefix+'.summ_stat.csv', sep=',')
    #                         param_row = pd.read_csv(prefix+'.param_row.csv', sep=',')
    #                         if self.tree_type == 'serial':
    #                             data = pd.read_csv(prefix+'.cblvs.csv', sep=',', header=None)
    #                         if self.tree_type == 'extant':
    #                             data = pd.read_csv(prefix+'.cdvs.csv', sep=',', header=None)
    #                         self.label_names = param_row.columns.to_list()
    #                         self.summ_stat_names = summ_stat.columns.to_list()
    #                         self.num_summ_stat = len(self.summ_stat_names)
    #                         self.num_labels = len(self.label_names)
    #                         self.num_data_row = int(data.shape[1] / size)
    #                         first_valid_file = False

    #         if self.tensor_format == 'csv':
    #             self.write_csv(size_sort)
    #         elif self.tensor_format == 'hdf5':
    #             self.write_hdf5(size_sort)