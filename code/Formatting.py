import os
import csv
import pandas as pd

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
        self.param_pred = args['param_pred'] # parameters in label set (prediction)
        self.param_data = args['param_data'] # parameters in data set (training, etc)
        self.in_dir     = self.sim_dir + '/' + self.job_name
        self.out_dir    = self.fmt_dir + '/' + self.job_name
        return

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        # collect files with replicate info
        print(self.in_dir)
        files = os.listdir(self.in_dir)
        info_files = [ x for x in files if 'info' in x ]

        print(info_files)

        # sort replicate indices into size-category lists
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

                if all_files_valid:
                    if size >= 0 and size not in size_sort:
                        size_sort[size] = []
                    if size >=0 and idx >= 0:
                        size_sort[size].append(idx)
                #else:
                #    print(all_files_valid,all_files)
        
        # build files
        for tree_size in sorted(list(size_sort.keys())):

            size_sort[tree_size].sort()

            print('Formatting {n} files for tree_type={tt} and tree_size={ts}'.format(n=len(size_sort[tree_size]), tt=self.tree_type, ts=tree_size))
            
            out_cblvs_fn  = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cblvs.data.csv'
            out_cdvs_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cdvs.data.csv'
            out_stat_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.summ_stat.csv'
            out_labels_fn = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.labels.csv'
            #out_info_fn   = self.out_dir + '/' + prefix + '.nt' + str(k) + '.info.csv'

            # cblvs tensor
            if self.tree_type == 'serial':
                with open(out_cblvs_fn, 'w') as outfile:
                    for i in size_sort[tree_size]:
                        fname = self.in_dir + '/' + 'sim.' + str(i) + '.cblvs.csv'
                        with open(fname, 'r') as infile:
                            s = infile.read()
                            z = outfile.write(s)
            # cdv file tensor       
            elif self.tree_type == 'extant':
                with open(out_cdvs_fn, 'w') as outfile:
                    for i in size_sort[tree_size]:
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

        # done
        return
