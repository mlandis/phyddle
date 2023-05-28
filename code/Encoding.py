# takes phylogeny, etc. and turns it into various CBLVS and CDVS formats

import Utilities
import numpy as np

class Encoder:
    def __init__(self, args, mdl): #, events_df, states_df):
        self.set_args(args)
        self.model = mdl
        return

    def set_args(self, args):
        # simulator arguments
        self.args              = args
        self.job_name          = args['job_name']
        self.model_name        = args['model_type']
        self.model_variant     = args['model_variant']
        self.sim_dir           = args['sim_dir']
        self.tree_sizes        = [ 200, 500 ]
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.rep_idx           = list(range(self.start_idx, self.end_idx))
        #self.tree_type         = args['tree_type']
        #self.model             = args['model']
        return
    
    def make_settings_str(self, idx, mtx_size):

        s = 'setting,value\n'
        s += 'job_name,' + self.job_name + '\n'
        s += 'model_type,' + self.model.model_type + '\n'
        s += 'model_variant,' + self.model.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'taxon_category,' + str(mtx_size) + '\n'
        return s

    def run(self):
        self.out_path  = self.sim_dir + '/' + self.job_name + '/sim'
        for idx in self.rep_idx:
            tmp_fn  = self.out_path + '.' + str(idx)
            self.encode_one(tmp_fn, idx)

    def encode_one(self, tmp_fn, idx):

        NUM_DIGITS = 10
        #np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')}, precision=NUM_DIGITS)
        #pd.set_option('display.precision', NUM_DIGITS)
        #pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

        # make filenames
        # out_path  = self.sim_dir + '/' + self.job_name + '/sim'
        geo_fn    = tmp_fn + '.geosse.nex'
        tre_fn    = tmp_fn + '.tre'
        prune_fn  = tmp_fn + '.extant.tre'
        nex_fn    = tmp_fn + '.nex'
        cblvs_fn  = tmp_fn + '.cblvs.csv'
        cdvs_fn   = tmp_fn + '.cdvs.csv'
        ss_fn     = tmp_fn + '.summ_stat.csv'
        info_fn   = tmp_fn + '.info.csv'
        #json_fn   = tmp_fn + '.json'
        #param1_fn = tmp_fn + '.param1.csv'
        #param2_fn = tmp_fn + '.param2.csv'

        # state space
        int2vec    = self.model.states.int2vec
        int2vecstr = self.model.states.int2vecstr #[ ''.join([str(y) for y in x]) for x in int2vec ]
        vecstr2int = self.model.states.vecstr2int #{ v:i for i,v in enumerate(int2vecstr) }

        # verify tree size & existence!
        #result_str     = ''
        n_taxa_idx     = Utilities.get_num_taxa(tre_fn) #, idx, self.tree_sizes)
        taxon_size_idx = Utilities.find_taxon_size(n_taxa_idx, self.tree_sizes)

        print(n_taxa_idx)
        print(taxon_size_idx)

        # handle simulation based on tree size
        if n_taxa_idx > np.max(self.tree_sizes):
            # too many taxa
            #result_str = '- replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)
            return #result_str
        elif n_taxa_idx <= 0:
            # no taxa
            #result_str = '- replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)
            return #result_str
        else:
            # valid number of taxa
            #result_str = '+ replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)

            # generate extinct-pruned tree
            prune_success = Utilities.make_prune_phy(tre_fn, prune_fn)

            # MJL 230411: probably too aggressive, should revisit
            #if not prune_success:
            #    next

            # generate nexus file 0/1 ranges
            taxon_states,nexus_str = Utilities.convert_nex(nex_fn, tre_fn, int2vec)
            Utilities.write_to_file(nexus_str, geo_fn)

            # then get CBLVS working
            cblv,new_order = Utilities.vectorize_tree(tre_fn, max_taxa=taxon_size_idx, prob=1.0 )
            cblvs = Utilities.make_cblvs_geosse(cblv, taxon_states, new_order)
        
            # NOTE: this if statement should not be needed, but for some reason the "next"
            # seems to run even when make_prune_phy returns False
            # generate CDVS file
            if prune_success:
                cdvs = Utilities.make_cdvs(prune_fn, taxon_size_idx, taxon_states, int2vecstr)
                #print(cdvs)

            # output files
            mtx_size = cblv.shape[1]


        # record info
        info_str = self.make_settings_str(idx, mtx_size)
        Utilities.write_to_file(info_str, info_fn)

        # record CBLVS data
        cblvs_str = np.array2string(cblvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
        cblvs_str = cblvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
        #cblvs_str = re.sub( '\.0+E\+0+', '', cblvs_str)
        #cblvs_str = Utilities.clean_scientific_notation(cblvs_str)
        #print(cblvs_str)
        Utilities.write_to_file(cblvs_str, cblvs_fn)

        # record CDVS data
        if prune_success:
            #cdvs = cdvs.to_numpy()
            cdvs_str = np.array2string(cdvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
            cdvs_str = cdvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
            #cdvs_str = Utilities.clean_scientific_notation(cdvs_str)
            Utilities.write_to_file(cdvs_str, cdvs_fn)

        # record summ stat data
        ss = Utilities.make_summ_stat(tre_fn, geo_fn, vecstr2int)
        ss_str = Utilities.make_summ_stat_str(ss)
        #ss_str = Utilities.clean_scientific_notation(ss_str) #re.sub( '\.0+E\+0+', '', ss_str)
        Utilities.write_to_file(ss_str, ss_fn)

        return