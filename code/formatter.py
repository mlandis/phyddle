import os
import csv

class Formatter:
    def __init__(self, fmt_args):
        self.set_args(fmt_args)
        return        

    def set_args(self, fmt_args):
        # simulator arguments
        self.fmt_args = fmt_args
        self.job_name = fmt_args['job_name']
        self.fmt_dir  = fmt_args['fmt_dir']
        self.sim_dir  = fmt_args['sim_dir']
        self.in_dir   = self.sim_dir + '/' + self.job_name
        self.out_dir  = self.fmt_dir + '/' + self.job_name
        return

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        # collect files with replicate info
        files = os.listdir(self.in_dir)
        info_files = [ x for x in files if 'info' in x ]

        # sort replicate indices into size-category lists
        size_sort = {}
        for fn in info_files:
            fn = self.in_dir + '/' + fn
            idx = -1
            size = -1
            with open(fn, newline='') as csvfile:
                info = csv.reader(csvfile, delimiter=',')
                for row in info:
                    if row[0] == 'replicate_index':
                        idx = int(row[1])
                    elif row[0] == 'taxon_category':
                        size = int(row[1])
                    #print(', '.join(row))

            if size >= 0 and size not in size_sort:
                size_sort[size] = []
            if size >=0 and idx >= 0:
                size_sort[size].append(idx)

        # build files
        for tree_size in sorted(list(size_sort.keys())):

            size_sort[tree_size].sort()

            print('Formatting {n} files for taxon_category={i}'.format(n=len(size_sort[tree_size]), i=tree_size))
            
            out_cdvs_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cdvs.data.csv'
            out_cblvs_fn  = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.cblvs.data.csv'
            out_stat_fn   = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.summ_stat.csv'
            out_labels_fn = self.out_dir + '/' + 'sim.nt' + str(tree_size) + '.labels.csv'
            #out_info_fn   = self.out_dir + '/' + prefix + '.nt' + str(k) + '.info.csv'
            
            # cdv file tensor
            with open(out_cdvs_fn, 'w') as outfile:
                for i in size_sort[tree_size]:
                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.cdvs.csv'
                    with open(fname, 'r') as infile:
                        s = infile.read()
                        z = outfile.write(s)

            # cblvs tensor
            with open(out_cblvs_fn, 'w') as outfile:
                for i in size_sort[tree_size]:
                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.cblvs.csv'
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
                    fname = self.in_dir + '/' + 'sim.' + str(i) + '.param2.csv'
                    with open(fname, 'r') as infile:
                        if j == 0:
                            s = infile.read()
                            z = outfile.write(s)
                        else:
                            s = ''.join(infile.readlines()[1:])
                            z = outfile.write(s)
    
        # done
        return
