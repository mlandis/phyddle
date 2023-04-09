from ete3 import Tree
import dendropy as dp
import numpy as np
from phyddle_util import *
import re

def make_summ_stat(tre_fn, geo_fn, states_bits_str_inv):
    
    # build summary stats
    summ_stats = {}

    # read tree + states
    phy = dp.Tree.get(path=tre_fn, schema="newick")
    num_taxa = len(phy.leaf_nodes())

    # tree statistics
    summ_stats['tree_length'] = phy.length()
    summ_stats['tree_height'] = max(phy.calc_node_ages())
    summ_stats['B1'] = dp.calculate.treemeasure.B1(phy)
    summ_stats['N_bar'] = dp.calculate.treemeasure.N_bar(phy)
    summ_stats['colless'] = dp.calculate.treemeasure.colless_tree_imbalance(phy)
    summ_stats['gamma'] = dp.calculate.treemeasure.pybus_harvey_gamma(phy)
    summ_stats['sackin'] = dp.calculate.treemeasure.sackin_index(phy)
    summ_stats['treeness'] = dp.calculate.treemeasure.treeness(phy)

    # read characters + states
    f = open(geo_fn, 'r')
    m = f.read().splitlines()
    f.close()
    y = re.search(string=m[2], pattern='NCHAR=([0-9]+)')
    z = re.search(string=m[3], pattern='SYMBOLS="([0-9A-Za-z]+)"')
    num_char = int(y.group(1))
    states = z.group(1)
    num_states = len(states)
    num_combo = num_char * num_states

    # get taxon data
    taxon_state_block = m[ m.index('Matrix')+1 : m.index('END;')-1 ]
    taxon_states = [ x.split(' ')[-1] for x in taxon_state_block ]

    # freqs of entire char-set
    freq_taxon_states = np.zeros(num_char, dtype='float')
    for i in range(num_char):
        summ_stats['char_' + str(i)] = 0
    for k in list(states_bits_str_inv.keys()):
        #freq_taxon_states[ states_bits_str_inv[k] ] = taxon_states.count(k) / num_taxa
        summ_stats['state_' + str(k)] = taxon_states.count(k) / num_taxa
        for i,j in enumerate(k):
            if j != '0':
                summ_stats['char_' + str(i)] += summ_stats['state_' + k]

    return summ_stats

def make_summ_stat_str(ss):
    keys_str = ','.join( list(ss.keys()) ) + '\n'
    vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
    return keys_str + vals_str

# for testing, to be removed
if False:
    fp = '/Users/mlandis/projects/phyddle/'
    dat_fp = fp + 'raw_data/bd1/'
    prefix = 'sim.62'
    tre_fn = dat_fp + prefix + '.tre'
    nex_fn = dat_fp + prefix + '.nex'
    geo_fn = dat_fp + prefix + '.geosse.nex'

    phy_nwk = dp.Tree.get(path=tre_fn, schema='newick')
    phy_nex = dp.Tree.get(path=nex_fn, schema='nexus')

    ss = make_summ_stat(tre_fn, geo_fn, states_bits_str_inv)
    ss_str = make_summ_stat_str(ss)