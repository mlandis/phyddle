# standard packages
import argparse
import importlib
import re
import os
import sys
import copy
from itertools import chain, combinations

# external packages
import pandas as pd
import numpy as np
import dendropy as dp


# Precision settings
NUM_DIGITS = 10
np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
pd.set_option('display.precision', NUM_DIGITS)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

##################
# Helper Classes #
##################

# model events
class Event:
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
        self.i = -1
        self.j = -1
        self.k = -1
        self.idx = idx
        if 'i' in idx:
            self.i = idx['i']
        if 'j' in idx:
            self.j = idx['j']
        if 'k' in idx:
            self.k = idx['k']
        self.rate = r
        self.name = n
        self.group = g
        self.ix = ix
        self.jx = jx
        self.reaction = ' + '.join(ix) + ' -> ' + ' + '.join(jx)
        
    # make print string
    def make_str(self):
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()


# state space
class States:
    def __init__(self, lbl2vec):

        # state space dictionary (input)
        self.lbl2vec      = lbl2vec

        # basic info
        self.int2lbl        = list( lbl2vec.keys() )
        self.int2vec        = list( lbl2vec.values() )
        self.int2int        = list( range(len(self.int2vec)) )
        self.int2set        = list( [ tuple([y for y,v in enumerate(x) if v == 1]) for x in self.int2vec ] )
        self.lbl_one        = list( set(''.join(self.int2lbl)) )
        self.num_char       = len( self.int2vec[0] )
        self.num_states     = len( self.lbl_one )

        # relational info
        self.lbl2int = {k:v for k,v in list(zip(self.int2lbl, self.int2int))}
        self.lbl2set = {k:v for k,v in list(zip(self.int2lbl, self.int2set))}
        self.lbl2vec = {k:v for k,v in list(zip(self.int2lbl, self.int2vec))}
        self.vec2int = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2int))}
        self.vec2lbl = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2lbl))}
        self.vec2set = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2set))}
        self.set2vec = {tuple(k):v for k,v in list(zip(self.int2set, self.int2vec))}
        self.set2int = {tuple(k):v for k,v in list(zip(self.int2set, self.int2int))}
        self.set2lbl = {tuple(k):v for k,v in list(zip(self.int2set, self.int2lbl))}
        self.int2vecstr = [ ''.join([str(y) for y in x]) for x in self.int2vec ]
        self.vecstr2int = { v:i for i,v in enumerate(self.int2vecstr) }
       
    def make_str(self):
        # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
        # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
        s = 'Statespace('
        x = []
        for i in self.int2int:
            # each state in the space is reported as STRING,INT,VECTOR;
            x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
        s += ';'.join(x) + ')'
        return s

    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()
    
    def make_df(self):
        df = pd.DataFrame()



#-----------------------------------------------------------------------------------------------------------------#

###################
# CONFIG LOADER   #
###################

def load_config(config_fn, arg_overwrite=True):
    
    # KEEP THIS: Want to improve precedence so CLI-provided-arg > CFG-arg > CLI-default-arg

    # # argument parsing
    # parser = argparse.ArgumentParser(description='phyddle pipeline config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, default='config', help='Config file name')
    # parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    # parser.add_argument('--proj',               dest='proj', type=str, default='my_project', help='Project name used as directory across pipeline stages')
    # parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, default=True, help='Use parallelization? (recommended)')
    # parser.add_argument('--num_proc',           dest='num_proc', type=int, default=-2, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)')
    # # directory settings
    # parser.add_argument('--sim_dir',            dest='sim_dir', type=str, default='../raw_data', help='Directory for raw simulated data')
    # parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, default='../tensor_data', help='Directory for tensor-formatted simulated data')
    # parser.add_argument('--net_dir',            dest='net_dir', type=str, default='../network', help='Directory for trained networks and predictions')
    # parser.add_argument('--plt_dir',            dest='plt_dir', type=str, default='../plot', help='Directory for plotted results')
    # parser.add_argument('--pred_dir',           dest='pred_dir', type=str, help='Predict results for dataset located in this directory')
    # # model settings
    # #parser.add_argument('--show_models',        dest='show_models', type=bool, default=False, help='Print all available model types and variants?')
    # parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type')
    # parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant')
    # parser.add_argument('--num_char',           dest='num_char', type=int, help='Number of characters')
    # # simulation settings
    # parser.add_argument('--sim_logging',        dest='sim_logging', type=str, default='verbose', choices=['clean', 'verbose', 'compress'], help='Simulation logging style')
    # parser.add_argument('--start_idx',          dest='start_idx', type=int, default=0, help='Start index for simulation')
    # parser.add_argument('--end_idx',            dest='end_idx', type=int, default=100, help='End index for simulation')
    # parser.add_argument('--stop_time',          dest='stop_time', type=float, default=10.0, help='Maximum duration of evolution for each simulation')
    # parser.add_argument('--stop_floor_sizes',   dest='stop_floor_sizes', type=int, default=0, help='Minimum number of taxa for each simulation')
    # parser.add_argument('--stop_ceil_sizes',    dest='stop_ceil_sizes', type=int, default=500, help='Maximum number of taxa for each simulation')
    # # formatting settings
    # parser.add_argument('--tensor_format',      dest='tensor_format', type=str, default='hdf5', choices=['hdf5', 'csv'], help='Storage format for simulation tensors')
    # parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree')
    # # learning settings
    # parser.add_argument('--tree_size',          dest='tree_size', type=int, help='Number of taxa in phylogenetic tensor')
    # parser.add_argument('--num_epochs',         dest='num_epochs', type=int, default=21, help='Number of learning epochs')
    # parser.add_argument('--batch_size',         dest='batch_size', type=int, default=128, help='Training batch sizes during learning')
    # parser.add_argument('--prop_test',          dest='prop_test', type=float, default=0.05, help='Proportion of data used as test examples (demonstrate trained network performance)')
    # parser.add_argument('--prop_validation',    dest='prop_validation', type=float, default=0.05, help='Proportion of data used as validation examples (diagnose network overtraining)')
    # parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, default=0.20, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)')
    # parser.add_argument('--alpha_CQRI',         dest='alpha_CQRI', type=float, default=0.95, help='Expected coverage percent for prediction intervals')
    # parser.add_argument('--loss',               dest='loss', type=str, default='mse', help='Loss function used as optimization criterion')
    # parser.add_argument('--optimizer',          dest='optimizer', type=str, default='adam', help='Method used for optimizing neural network')
    # # plotting settings
    # parser.add_argument('--network_prefix',     dest='network_prefix', type=str, help='Plot results related to this network prefix')
    # # prediction settings
    # parser.add_argument('--pred_prefix',        dest='pred_prefix', type=str, help='Predict results for this dataset')
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config') #,
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, help='Config file name')
    #parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    parser.add_argument('-p', '--proj',         dest='proj', type=str, help='Project name used as directory across pipeline stages')
    parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, help='Use parallelization? (recommended)')
    parser.add_argument('--num_proc',           dest='num_proc', type=int, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)')
    # directory settings
    parser.add_argument('--sim_dir',            dest='sim_dir', type=str, help='Directory for raw simulated data')
    parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, help='Directory for tensor-formatted simulated data')
    parser.add_argument('--net_dir',            dest='net_dir', type=str, help='Directory for trained networks and predictions')
    parser.add_argument('--plt_dir',            dest='plt_dir', type=str, help='Directory for plotted results')
    parser.add_argument('--pred_dir',           dest='pred_dir', type=str, help='Predict results for dataset located in this directory')
    # model settings
    #parser.add_argument('--show_models',        dest='show_models', type=bool, help='Print all available model types and variants?')
    parser.add_argument('--show_models',        action='store_true', help='Print all available model types and variants?')
    parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type')
    parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant')
    parser.add_argument('--num_char',           dest='num_char', type=int, help='Number of characters')
    # simulation settings
    parser.add_argument('--sim_logging',        dest='sim_logging', type=str, choices=['clean', 'verbose', 'compress'], help='Simulation logging style')
    parser.add_argument('--start_idx',          dest='start_idx', type=int, help='Start index for simulation')
    parser.add_argument('--end_idx',            dest='end_idx', type=int, help='End index for simulation')
    parser.add_argument('--stop_time',          dest='stop_time', type=float, help='Maximum duration of evolution for each simulation')
    parser.add_argument('--min_num_taxa',       dest='min_num_taxa', type=int, help='Minimum number of taxa for each simulation')
    parser.add_argument('--max_num_taxa',       dest='max_num_taxa', type=int, help='Maximum number of taxa for each simulation')
    # formatting settings
    parser.add_argument('--tensor_format',      dest='tensor_format', type=str, choices=['hdf5', 'csv'], help='Storage format for simulation tensors')
    parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree')
    parser.add_argument('--save_phyenc_csv',    dest='save_phyenc_csv', type=bool, help='Save encoded phylogenetic tensor encoding to csv?')
    # learning settings
    parser.add_argument('--tree_size',          dest='tree_size', type=int, help='Number of taxa in phylogenetic tensor')
    parser.add_argument('--num_epochs',         dest='num_epochs', type=int, help='Number of learning epochs')
    parser.add_argument('--batch_size',         dest='batch_size', type=int, help='Training batch sizes during learning')
    parser.add_argument('--prop_test',          dest='prop_test', type=float, help='Proportion of data used as test examples (demonstrate trained network performance)')
    parser.add_argument('--prop_validation',    dest='prop_validation', type=float, help='Proportion of data used as validation examples (diagnose network overtraining)')
    parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)')
    parser.add_argument('--cpi_coverage',       dest='cpi_coverage', type=float, help='Expected coverage percent for calibrated prediction intervals')
    parser.add_argument('--loss',               dest='loss', type=str, help='Loss function used as optimization criterion')
    parser.add_argument('--optimizer',          dest='optimizer', type=str, help='Method used for optimizing neural network')
    # plotting settings
    #parser.add_argument('--network_prefix',     dest='network_prefix', type=str, help='Plot results related to this network prefix')
    # prediction settings
    parser.add_argument('--pred_prefix',        dest='pred_prefix', type=str, help='Predict results for this dataset')
    parser.add_argument('--plot_train_color',    dest='plot_train_color', type=str, help='Plotting color for training data elements')
    parser.add_argument('--plot_label_color',    dest='plot_label_color', type=str, help='Plotting color for training label elements')
    parser.add_argument('--plot_test_color',     dest='plot_test_color', type=str, help='Plotting color for test data elements')
    parser.add_argument('--plot_validation_color', dest='plot_validation_color', type=str, help='Plotting color for validation data elements')
    parser.add_argument('--plot_aux_data_color', dest='plot_aux_data_color', type=str, help='Plotting color for auxiliary input data elements')
    parser.add_argument('--plot_pred_color',     dest='plot_pred_color', type=str, help='Plotting color for prediction data elements')


    # parse arguments
    args = parser.parse_args()
    
    # print models & exit
    if args.show_models:
         print(args.show_models)
         import ModelLoader
         model_str = ModelLoader.make_model_registry_str()
         print(model_str)
         sys.exit()

    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn != None:
        config_fn = args.config_fn
    config_fn = config_fn.rstrip('.py')

    # config from file
    m = importlib.import_module(config_fn)

    def overwrite_defaults(m, args, var):
        x = getattr(args, var)
        if x is not None:
            # if args.force == True:
            #     m.args[var] = x
            # elif var not in m.args:
            #     m.args[var] = x
            m.args[var] = x
        return m
    
    # update arguments from defaults, when provided
    m = overwrite_defaults(m, args, 'proj')
    m = overwrite_defaults(m, args, 'use_parallel')
    m = overwrite_defaults(m, args, 'num_proc')
    m = overwrite_defaults(m, args, 'sim_dir')
    m = overwrite_defaults(m, args, 'fmt_dir')
    m = overwrite_defaults(m, args, 'net_dir')
    m = overwrite_defaults(m, args, 'plt_dir')
    m = overwrite_defaults(m, args, 'pred_dir')
    #m = overwrite_defaults(m, args, 'show_models')
    m = overwrite_defaults(m, args, 'model_type')
    m = overwrite_defaults(m, args, 'model_variant')
    m = overwrite_defaults(m, args, 'num_char')
    m = overwrite_defaults(m, args, 'sim_logging')
    m = overwrite_defaults(m, args, 'start_idx')
    m = overwrite_defaults(m, args, 'end_idx')
    m = overwrite_defaults(m, args, 'stop_time')
    m = overwrite_defaults(m, args, 'min_num_taxa')
    m = overwrite_defaults(m, args, 'max_num_taxa')
    m = overwrite_defaults(m, args, 'tree_size')
    m = overwrite_defaults(m, args, 'save_phyenc_csv')
    m = overwrite_defaults(m, args, 'num_epochs')
    m = overwrite_defaults(m, args, 'batch_size')
    m = overwrite_defaults(m, args, 'prop_test')
    m = overwrite_defaults(m, args, 'prop_validation')
    m = overwrite_defaults(m, args, 'prop_calibration')
    m = overwrite_defaults(m, args, 'cpi_coverage')
    m = overwrite_defaults(m, args, 'loss')
    m = overwrite_defaults(m, args, 'optimizer')
    #m = overwrite_defaults(m, args, 'network_prefix')
    m = overwrite_defaults(m, args, 'pred_prefix')
    m = overwrite_defaults(m, args, 'plot_train_color')
    m = overwrite_defaults(m, args, 'plot_test_color')
    m = overwrite_defaults(m, args, 'plot_validation_color')
    m = overwrite_defaults(m, args, 'plot_aux_data_color')
    m = overwrite_defaults(m, args, 'plot_label_color')
    m = overwrite_defaults(m, args, 'plot_pred_color')         

    # return new args
    return m.args


#-----------------------------------------------------------------------------------------------------------------#

#########################
# Model registry        #
#########################

# def show_models(args):

#     model_variants = {
#         'GeoSSE' : {
#             'variants': {
#                 'equal-rates',
#                 'free-rates',
#                 'density-extinction'
#             },
                    
#         'SIRM'   : { 'equal-rates', 'free-rates' }
#     }
#     model_
#     cw = [20, 20, 40]
#     s  = 'Model type'.ljust(cw[0], ' ')  + 'Model variant'.ljust(cw[1], ' ') + 'Parameters'.ljust(cw[2], ' ') + '\n'
#     s += ''.ljust(sum(cw), '-') + '\n'
#     for i,model_type in enumerate(ModelLoader.model_type_list):
#         s += model_type.ljust(cw[0], ' ')
#         model = ModelLoader.load_model(model_type)
#         # for j,model_variant in enumerate(model.get_model_variants()):
#         #     if j == 0:
#         #         s += '' + model_variant.ljust(cw[1], ' ')
#         #     else:
#         #         s += ''.ljust(cw[0], ' ')  + model_variant.ljust(cw[1], ' ')
#     return s



#-----------------------------------------------------------------------------------------------------------------#

###################
# GENERAL HELPERS #
###################

def make_symm(m):
    d = np.diag(m)
    m = np.triu(m)
    m = m + m.T
    np.fill_diagonal(m, d)
    return m

# Chat-GPT function
def sort_binary_vectors(binary_vectors):
    """
    Sorts a list of binary vectors in order of number of "on" bits first, and then left to right in terms of which bits are "on".
    """
    # Define a helper function to count the number of "on" bits in a binary vector
    def count_ones(binary_vector):
        return sum(binary_vector)
    
    # Sort the binary vectors in the list first by number of "on" bits
    sorted_vectors = sorted(binary_vectors, key=count_ones)
    
    # Sort the binary vectors in the list by "on" bits from left to right
    for i in range(len(sorted_vectors)):
        for j in range(i+1, len(sorted_vectors)):
            if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
                for k in range(len(sorted_vectors[i])):
                    if sorted_vectors[i][k] != sorted_vectors[j][k]:
                        if sorted_vectors[j][k] > sorted_vectors[i][k]:
                            sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
                        break
                
    return sorted_vectors


# helper functions
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def find_tree_width(num_taxa, max_taxa):
    if num_taxa == 0:
        return 0
    elif num_taxa > max_taxa[-1]:
        return -1
    for i in max_taxa:
        if num_taxa <= i:
            return i
    # should never call this
    raise Exception('error in find_tree_width()', num_taxa, max_taxa)
    return -2



#-----------------------------------------------------------------------------------------------------------------#


################
# FILE HELPERS #
################

def write_to_file(s, fn):
    f = open(fn, 'w')
    f.write(s)
    f.close()

def read_tree(tre_fn):
    # check that file exists 
    if not os.path.exists(tre_fn):
        raise FileNotFoundError(f'Could not find tree file at {tre_fn}')
    
    phy = None
    for schema in [ 'newick', 'nexus' ]:
        try:
            phy_tmp = dp.Tree.get(path=tre_fn, schema=schema)
        except:
            phy_tmp = None
        else:
             if phy_tmp is not None:
                phy = phy_tmp
    return phy




#-----------------------------------------------------------------------------------------------------------------#

#####################
# FORMAT CONVERTERS #
#####################

def convert_nexus_to_array(dat_fn):
    
    # read file
    f = open(dat_fn, 'r')
    lines = f.readlines()
    f.close()

    # process file
    found_matrix = False
    num_taxa    = 0
    num_char    = 0
    taxon_idx   = 0
    taxon_names = []
    for line in lines:
        # purge whitespace
        line = ' '.join(line.split()).rstrip('\n')
        tok = line.split(' ')
        
        # skip lines with comments
        if tok[0] == '[':
            continue

        # get data dimenions
        if tok[0].upper() == 'DIMENSIONS':
            for x in tok:
                x = x.rstrip(';')
                if 'NTAX' in x.upper():
                    num_taxa = int(x.split('=')[1])
                elif 'NCHAR' in x.upper():
                    num_char = int(x.split('=')[1])
            dat = np.zeros((num_char, num_taxa), dtype='int')

        # entering data matrix
        if tok[0].upper() == 'MATRIX':
            found_matrix = True
            continue

        # process data matrix
        if found_matrix:
            if tok[0] == ';':
                found_matrix = False
                break
            elif len(tok) == 2:
                #print(tok)
                name = tok[0]
                state = tok[1]
                taxon_names.append(name)
                dat[:,taxon_idx] = [ int(z) for z in state ]
                taxon_idx += 1

    # construct data frame
    # rows: char states
    # cols: taxa
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df


def convert_table_to_array(dat_fn, sep=","):
    
    # read file
    f = open(dat_fn, 'r')
    lines = f.readlines()
    f.close()

    # process file
    num_taxa    = len(lines)
    num_char    = 0
    taxon_idx   = 0
    taxon_names = []
    first_taxon = True

    for line in lines:
        # purge whitespace
        line = ' '.join(line.split()).rstrip('\n')
        tok = line.split(sep)
        
        # get taxon + state
        name = tok[0]
        state = tok[1]

        # construct matrix based on num char
        if first_taxon:
            first_taxon = False
            num_char = len(state)
            dat = np.zeros((num_char, num_taxa), dtype='int')

        # save taxon name, populate array
        taxon_names.append(name)
        dat[:,taxon_idx] = [ int(z) for z in state ]
        taxon_idx += 1

    # construct data frame
    # rows: char states
    # cols: taxa
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df


# Converts MASTER output into nex
# move to MasterSimulator?
def convert_phy2dat_nex(phy_nex_fn, int2vec):

    # get num regions from size of bit vector
    num_char = len(int2vec[0])

    # get tip names and states from NHX tree
    nex_file = open(phy_nex_fn, 'r')
    nex_str  = nex_file.readlines()[3]
    matches  = re.findall(pattern='([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"', string=nex_str)
    num_taxa = len(matches)
    nex_file.close()

    # generate taxon-state data
    #d = {}
    s_state_str = ''
    for i,v in enumerate(matches):
        taxon        = v[0]
        state        = int(v[2])
        vec_str      = ''.join([ str(x) for x in int2vec[state] ])
        #d[ taxon ]   = vec_str
        s_state_str += taxon + '  ' + vec_str + '\n'
    
    # build new nexus string
    s = \
'''#NEXUS
Begin DATA;
Dimensions NTAX={num_taxa} NCHAR={num_char}
Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
Matrix
{s_state_str}
;
END;
'''.format(num_taxa=num_taxa, num_char=num_char, s_state_str=s_state_str)

    return s

## set return None if bad, then flag the index as a bad sim.
#def make_prune_phy(tre_fn, prune_fn):
def make_prune_phy(phy, prune_fn):
    # read tree
    # phy_ = dp.Tree.get(path=tre_fn, schema='newick')
    # copy input tree
    phy_ = copy.deepcopy(phy)
    # compute all root-to-node distances
    root_distances = phy_.calc_node_root_distances()
    # find tree height (max root-to-node distance)
    tree_height = np.max( root_distances )
    # create empty dictionary
    d = {}
    # loop through all leaf nodes
    leaf_nodes = phy_.leaf_nodes()
    for i,nd in enumerate(leaf_nodes):
        # convert root-distances to ages
        age = tree_height - nd.root_distance
        nd.annotations.add_new('age', age)
        # ultrametricize ages for extant taxa
        if age < 1e-6:
            age = 0.0
        # store taxon and age in dictionary
        taxon_name = str(nd.taxon).strip('\'')
        taxon_name = taxon_name.replace(' ', '_')
        d[ taxon_name ] = age
    # determine what to drop
    drop_taxon_labels = [ k for k,v in d.items() if v > 1e-12 ]
    # inform user if pruning yields valid tree
    if len(leaf_nodes) - len(drop_taxon_labels) < 2:
        return None
    else:
        # prune non-extant taxa
        phy_.prune_taxa_with_labels( drop_taxon_labels )
        # write pruned tree
        phy_.write(path=prune_fn, schema='newick')
        return phy_



# Used in Encoding
def settings_to_str(settings, taxon_category):
    s = 'setting,value\n'
    s += 'model_name,' + settings['model_name'] + '\n'
    s += 'model_type,' + settings['model_type'] + '\n'
    s += 'replicate_index,' + str(settings['replicate_index']) + '\n'
    s += 'taxon_category,' + str(taxon_category) + '\n'
    return s

def param_dict_to_str(params):
    s1 = 'param,i,j,value\n'
    s2 = ''
    s3 = ''
    for k,v in params.items():
        for i,x in enumerate(v):
            if len(v.shape) == 1:
                rate = np.round(x, NUM_DIGITS)
                s1 += '{k},{i},{i},{v}\n'.format(k=k,i=i,v=rate)
                s2 += '{k}_{i},'.format(k=k,i=i)
                s3 += str(rate) + ','
            else:
                for j,y in enumerate(x):
                    rate = np.round(y, NUM_DIGITS)
                    s1 += '{k},{i},{j},{v}\n'.format(k=k,i=i,j=j,v=rate)
                    s2 += '{k}_{i}_{j},'.format(k=k,i=i,j=j)
                    s3 += str(rate) + ','

    s4 = s2.rstrip(',') + '\n' + s3.rstrip(',') + '\n'
    return s1,s4

def events2df(events):
    df = pd.DataFrame({
        'name'     : [ e.name for e in events ],
        'group'    : [ e.group for e in events ], 
        'i'        : [ e.i for e in events ],
        'j'        : [ e.j for e in events ],
        'k'        : [ e.k for e in events ],
        'reaction' : [ e.reaction for e in events ],
        'rate'     : [ e.rate for e in events ]
    })
    return df

def states2df(states):
    df = pd.DataFrame({
        'lbl' : states.int2lbl,
        'int' : states.int2int,
        'set' : states.int2set,
        'vec' : states.int2vec
    })
    return df

# make matrix with parameter values, lower-bounds, upper-bounds: 3D->2D
def make_param_VLU_mtx(A, param_names):
    
    # axis labels
    stat_names = ['value', 'lower', 'upper']

    # multiindex
    index = pd.MultiIndex.from_product([range(s) for s in A.shape], names=['stat', 'rep_idx', 'param'])
    
    # flattened data frame
    df = pd.DataFrame({'A': A.flatten()}, index=index)['A']
    df = df.reorder_levels(['param','stat','rep_idx']).sort_index()

    # unstack stat and param, so they become combined header indices
    df = df.unstack(level=['stat','param'])
    #col_names = df.columns

    new_col_names = [ f'{param_names[y]}_{stat_names[x]}' for x,y in df.columns ]
    df.columns = new_col_names

    return df

def make_clean_phyloenc_str(x):
    s = np.array2string(x, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)
    s = s + '\n'
    return s

def clean_scientific_notation(s):
    return re.sub( '\.0+E\+0+', '', s)



#-----------------------------------------------------------------------------------------------------------------#

#########################
# Tensor de/normalizing #
#########################

def normalize(data, m_sd = None):
    if(type(m_sd) == type(None)):
        m = data.mean(axis = 0)
        sd = data.std(axis = 0)
        sd[np.where(sd == 0)] = 1
        return (data - m)/sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0])/m_sd[1]
        
    
def denormalize(data, train_mean, train_sd, log_labels = False):
    return data * train_sd + train_mean

#-----------------------------------------------------------------------------------------------------------------#






#-----------------------------------------------------------------------------------------------------------------#


# #######################
# # CQR functions      ##
# #######################

# # ==> Move to Learning? <==

# def pinball_loss(y_true, y_pred, alpha):
#     err = y_true - y_pred
#     return K.mean(K.maximum(alpha*err, (alpha-1)*err), axis=-1)

# def pinball_loss_q_0_025(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.025)

# def pinball_loss_q_0_975(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.975)

# def pinball_loss_q_0_05(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.05)

# def pinball_loss_q_0_95(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.95)

# def pinball_loss_q_0_10(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.10)

# def pinball_loss_q_0_90(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.90)

# def pinball_loss_q_0_15(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.15)

# def pinball_loss_q_0_85(y_true, y_pred):
#     return pinball_loss(y_true, y_pred, alpha=0.85)

# # computes the distance y_i is inside/outside the lower(x_i) and upper(x_i) quantiles
# # there are three cases to consider:
# #   1. y_i is under the lower bound: max-value will be q_lower(x_i) - y_i & positive
# #   2. y_i is over the upper bound:  max-value will be y_i - q_upper(x_i) & positive
# #   3. y_i is between the bounds:    max-value will be the difference between y_i and the closest bound & negative
# def compute_conformity_scores(x, y, q_lower, q_upper):
#     return np.max( q_lower(x)-y, y-q_upper(x) )

# def get_CQR_constant(preds, true, inner_quantile=0.95, symmetric = True):
#     #preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the params
#     # compute non-comformity scores
#     Q = np.empty((2, preds.shape[2]))
    
#     for i in range(preds.shape[2]):
#         if symmetric:
#             # Symmetric non-comformity score
#             s = np.amax(np.array((preds[0][:,i] - true[:,i], true[:,i] - preds[1][:,i])), axis=0)
#             # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
#             #Q = np.append(Q, np.quantile(s, inner_quantile * (1 + 1/preds.shape[1])))
#             lower_q = np.quantile(s, inner_quantile * (1 + 1/preds.shape[1]))
#             upper_q = lower_q
#             #Q[:,i] = np.array([lower_q, upper_q])
#         else:
#             # Asymmetric non-comformity score
#             lower_s = np.array(true[:,i] - preds[0][:,i])
#             upper_s = np.array(true[:,i] - preds[1][:,i])
#             lower_q = np.quantile(lower_s, (1 - inner_quantile)/2 * (1 + 1/preds.shape[1]))
#             upper_q = np.quantile(upper_s, (1 + inner_quantile)/2 * (1 + 1/preds.shape[1]))
#             # get (lower_q adjustment, upper_q adjustment)

#         Q[:,i] = np.array([lower_q, upper_q])
                               
#     return Q

# def get_CQR_constant(x_pred_quantiles, y_true, inner_quantile=0.95):
#     # preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the param label
#     # compute non-comformity scores
#     Q = np.array([])
#     # error tolerance on quantile for E
#     error = 0.001
#     # for each parameter
#     #inner_quantile * (1 + 1/x_pred_quantiles.shape[1])
#     for i in range(x_pred_quantiles.shape[2]):
#         E = np.amax(np.array((x_pred_quantiles[0][:,i] - y_true[:,i], y_true[:,i] - x_pred_quantiles[1][:,i])), axis=0)

#         # get 1 - alpha/2's quintile of non-comformity scores
#         #print( inner_quantile * (1 + 1/x_pred_quantiles.shape[1]) )
#         quant = inner_quantile * (1 + 1/x_pred_quantiles.shape[1])
#         # if quant < 0 and quant > 0 - error:
#         #     quant = 0.
#         # elif quant > 1. and quant < 1. + error:
#         #     quant = 1.
#         Q = np.append(Q, np.quantile(E, quant))

#     return Q
# def get_CQR_constant_old(preds, true, inner_quantile=0.95, symmetric = True):
#     #preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the params
#     # compute non-comformity scores
#     Q = np.array([]) if symmetric else np.empty((2, preds.shape[2]))
#     for i in range(preds.shape[2]):
#         if symmetric:
#             # Symmetric non-comformity score
#             s = np.amax(np.array((preds[0][:,i] - true[:,i], true[:,i] - preds[1][:,i])), axis=0)
#             # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
#             Q = np.append(Q, np.quantile(s, inner_quantile * (1 + 1/preds.shape[1])))
#         else:
#             # Asymmetric non-comformity score
#             lower_s = np.array(true[:,i] - preds[0][:,i])
#             upper_s = np.array(true[:,i] - preds[1][:,i])
#             # get (lower_q adjustment, upper_q adjustment)
#             Q[:,i] = np.array((np.quantile(lower_s, (1 - inner_quantile)/2 * (1 + 1/preds.shape[1])),
#                                np.quantile(upper_s, (1 + inner_quantile)/2 * (1 + 1/preds.shape[1]))))
#     return Q

