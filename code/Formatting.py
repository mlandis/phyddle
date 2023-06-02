import os
import csv
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

class InputFormatter:
    def __init__(self, args):
        self.set_args(args)
        return        

    def set_args(self, args):
        # simulator arguments
        self.args       = args
        self.job_name   = args['job_name']
        self.fmt_dir    = args['fmt_dir']
        self.sim_dir    = args['sim_dir']
        self.tree_type  = args['tree_type']
        self.num_char   = args['num_char']
        self.param_pred = args['param_pred'] # parameters in label set (prediction)
        self.param_data = args['param_data'] # parameters in data set (training, etc)
        self.tensor_format = args['tensor_format']
        self.in_dir     = self.sim_dir + '/' + self.job_name
        self.out_dir    = self.fmt_dir + '/' + self.job_name
        return

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        # collect files with replicate info
        files = os.listdir(self.in_dir)
        info_files = [ x for x in files if 'info' in x ]

        # sort replicate indices into size-category lists
        first_valid_file = True
        size_sort = {}
        for fn in info_files:
            fn = self.sim_dir + '/' + self.job_name + '/' + fn
            idx = -1
            size = -1
            all_files_valid = False

            with open(fn, newline='') as csvfile:
                info = csv.reader(csvfile, delimiter=',')
                for row in info:
                    if row[0] == 'replicate_index':
                        idx = int(row[1])
                    elif row[0] == 'taxon_category':
                        size = int(row[1])
                    
                # check that all necessary files exist
                #all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cdvs.csv','cblvs.csv','param2.csv','summ_stat.csv']]
                if self.tree_type == 'serial':
                    all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cblvs.csv','param_row.csv','summ_stat.csv']]
                elif self.tree_type == 'extant':
                    all_files = [self.in_dir+'/sim.'+str(idx)+'.'+x for x in ['cdvs.csv','param_row.csv','summ_stat.csv']]
                else:
                    raise NotImplementedError
                all_files_valid = all( [os.path.isfile(fn) for fn in all_files] )

                # place index into tree_size category if all necessary files exist
                if all_files_valid:
                    if size >= 0 and size not in size_sort:
                        size_sort[size] = []
                    if size >=0 and idx >= 0:
                        size_sort[size].append(idx)

                    # collect simpler header info from summ_stat and param_row 
                    if first_valid_file:
                        prefix = f'{self.in_dir}/sim.{idx}'
                        summ_stat = pd.read_csv(prefix+'.summ_stat.csv', sep=',')
                        param_row = pd.read_csv(prefix+'.param_row.csv', sep=',')
                        if self.tree_type == 'serial':
                            data = pd.read_csv(prefix+'.cblvs.csv', sep=',', header=None)
                        if self.tree_type == 'extant':
                            data = pd.read_csv(prefix+'.cdvs.csv', sep=',', header=None)
                        self.label_names = param_row.columns.to_list()
                        self.summ_stat_names = summ_stat.columns.to_list()
                        self.num_summ_stat = len(self.summ_stat_names)
                        self.num_labels = len(self.label_names)
                        self.num_data_row = int(data.shape[1] / size)
                        first_valid_file = False

        if self.tensor_format == 'csv':
            self.write_csv(size_sort)
        elif self.tensor_format == 'hdf5':
            self.write_hdf5(size_sort)

    def write_hdf5(self, size_sort):
                # build files
        for tree_size in sorted(list(size_sort.keys())):
            # dimensions
            size_sort[tree_size].sort()
            num_samples = len(size_sort[tree_size])
            num_taxa = tree_size
            num_data_length = num_taxa * self.num_data_row
            
            print('Formatting {n} files for tree_type={tt} and tree_size={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_size))
            
            # HDF5 file
            out_hdf5_fn  = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.hdf5'
            hdf5_file  = h5py.File(out_hdf5_fn, 'w')

            self.summ_stat_names_encode = [ s.encode('UTF-8') for s in self.summ_stat_names ]
            self.label_names_encode = [ s.encode('UTF-8') for s in self.label_names ]
            dat_stat_names = hdf5_file.create_dataset('summ_stat_names', (1, self.num_summ_stat), 'S64', self.summ_stat_names_encode)
            dat_label_names = hdf5_file.create_dataset('label_names', (1, self.num_labels), 'S64', self.label_names_encode)

            # numerical data
            dat_data   = hdf5_file.create_dataset('data', (num_samples, num_data_length), dtype='f', compression='gzip')
            dat_stat   = hdf5_file.create_dataset('summ_stat', (num_samples, self.num_summ_stat), dtype='f', compression='gzip')
            dat_labels = hdf5_file.create_dataset('labels', (num_samples, self.num_labels), dtype='f', compression='gzip')
      
            # save in chunks of 10000
            # this is somewhat slow, but could be sped up by reading and saving in chunks
            # ... nevermind, seems like bottleneck was pandas not h5py
            #chunk_size = 1000
            #df_data_chunk = np.empty( (chunk_size, num_data_length) )
            #df_stat_chunk = np.empty( (chunk_size, self.num_summ_stat) )
            #df_labels_chunk = np.empty( (chunk_size, self.num_labels) )

            for j,i in enumerate(size_sort[tree_size]):
                
                fname_base = f'{self.in_dir}/sim.{i}'
                if self.tree_type == 'serial':
                    fname_data = fname_base + '.cblvs.csv'
                elif self.tree_type == 'extant':
                    fname_data = fname_base + '.cdvs.csv'
                fname_param = fname_base + '.param_row.csv'
                fname_stat = fname_base + '.summ_stat.csv'

                #chunk_idx = j % chunk_size
                #save_chunk = ((chunk_idx+1) == 0)
                #print(j, chunk_idx, save_chunk)

                #df = pd.read_csv(fname_data, header=None)
                #df_data_chunk[chunk_idx,:] = df.to_numpy()   
                #df = pd.read_csv(fname_stat)
                #df_stat_chunk[chunk_idx,:] = df.to_numpy()
                #df = pd.read_csv(fname_param)
                #df_labels_chunk[chunk_idx,:] = df.to_numpy()

                #df_data_chunk[chunk_idx,:] = np.loadtxt(fname_data, delimiter=',') # df.to_numpy()   
                #df_stat_chunk[chunk_idx,:] = np.loadtxt(fname_stat, delimiter=',', skiprows=1) #df.to_numpy()
                #df_labels_chunk[chunk_idx,:] = np.loadtxt(fname_param, delimiter=',', skiprows=1) #df.to_numpy()

                dat_data[j,:] = np.loadtxt(fname_data, delimiter=',') # df.to_numpy()   
                dat_stat[j,:] = np.loadtxt(fname_stat, delimiter=',', skiprows=1) #df.to_numpy()
                dat_labels[j,:] = np.loadtxt(fname_param, delimiter=',', skiprows=1) #df.to_numpy()
                # save chunk
                #if save_chunk:
                #    start_chunk_idx = j-chunk_size
                #    end_chunk_idx = j
                #    print('save chunk ', start_chunk_idx)
                #    dat_data[start_chunk_idx:end_chunk_idx,:] = df_data_chunk
                #    dat_stat[start_chunk_idx:end_chunk_idx,:] = df_stat_chunk
                #    dat_labels[start_chunk_idx:end_chunk_idx,:] = df_labels_chunk

#            # cdv file tensor       
#            elif self.tree_type == 'extant':
#                for j,i in enumerate(size_sort[tree_size]):
#                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.cdvs.csv'
#                    df = pd.read_csv(fname, header=None)
#                    dat_data[j,:] = df.to_numpy()
#                    
#            # summary stats tensor
#            for j,i in enumerate(size_sort[tree_size]):
#                fname = self.in_dir + '/' + 'sim.' + str(i) + '.summ_stat.csv'
#                df = pd.read_csv(fname)
#                dat_stat[j,:] = df.to_numpy()
#            
#            # labels input tensor
#            for j,i in enumerate(size_sort[tree_size]):
#                fname = self.in_dir + '/' + 'sim.' + str(i) + '.param_row.csv'
#                df = pd.read_csv(fname)
#                dat_labels[j,:] = df.to_numpy()
            
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

    def write_csv(self, size_sort):
        # build files
        for tree_size in sorted(list(size_sort.keys())):
            # dimensions
            size_sort[tree_size].sort()
            num_samples = len(size_sort[tree_size])
            num_taxa = tree_size
            num_data_length = num_taxa * self.num_data_row
            
            print('Formatting {n} files for tree_type={tt} and tree_size={ts}'.format(n=num_samples, tt=self.tree_type, ts=tree_size))
            
            # CSV files
            out_cblvs_fn  = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cblvs.data.csv'
            out_cdvs_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cdvs.data.csv'
            out_stat_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.summ_stat.csv'
            out_labels_fn = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.labels.csv'

            # cblvs tensor
            if self.tree_type == 'serial':
                with open(out_cblvs_fn, 'w') as outfile:
                    for j,i in enumerate(size_sort[tree_size]):
                        fname = self.in_dir + '/' + 'sim.' + str(i) + '.cblvs.csv'
                        with open(fname, 'r') as infile:
                            s = infile.read()
                            z = outfile.write(s)
                        

            # cdv file tensor       
            elif self.tree_type == 'extant':
                with open(out_cdvs_fn, 'w') as outfile:
                    for j,i in enumerate(size_sort[tree_size]):
                        fname = self.in_dir + '/' + 'sim.' + str(i) + '.cdvs.csv'
                        with open(fname, 'r') as infile:
                            s = infile.read()
                            z = outfile.write(s)
                    
            # summary stats tensor
            with open(out_stat_fn, 'w') as outfile:
                for j,i in enumerate(size_sort[tree_size]):
                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.summ_stat.csv'
                    with open(fname, 'r') as infile:
                        if j == 0:
                            s = infile.read()
                            z = outfile.write(s)
                        else:
                            s = ''.join(infile.readlines()[1:])
                            z = outfile.write(s)
                        

            # labels input tensor
            with open(out_labels_fn, 'w') as outfile:
                for j,i in enumerate(size_sort[tree_size]):
                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.param_row.csv'
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
