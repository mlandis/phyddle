#!/usr/local/bin/python3

import joblib
import time
import pandas as pd
import numpy as np
import scipy as sp
import dendropy as dp
import random
import re
import sys
import argparse

from collections import Counter
from itertools import chain, combinations
from ete3 import Tree

sys.setrecursionlimit(100000) # possibly needed for CBLV formatting

#import dendropy as dp
#import itertools
#import ammon_tree_utilities as tu #from phylodeep import tree_utilities as tu
#import ammon_encoding as en #from phylodeep import encoding as en

def add_dist_to_root(tre):
    """
    Add distance to root (dist_to_root) attribute to each node
    :param tre: ete3.Tree, tree on which the dist_to_root should be added
    :return: void, modifies the original tree
    """

    for node in tre.traverse("preorder"):
        if node.is_root():
            node.add_feature("dist_to_root", 0)
        elif node.is_leaf():
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
        else:
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
    return None


def name_tree(tre, newLeafKeys_inputNameValues):
    """
    Names all the tree nodes that are not named, with unique names.
    :param tre: ete3.Tree, the tree to be named
    :return: void, modifies the original tree
    """
    existing_names = Counter((_.name for _ in tre.traverse() if _.name))
    
    i = 0
    for node in tre.traverse('levelorder'):
        if(node.is_leaf()): # A.M.T
        	#new_leaf_order_names.append((i, node.name))
                newLeafKeys_inputNameValues[i] = node.name
        node.name = i
        i += 1
   
    return None


def rescale_tree(tre, target_avg_length):
    """
    Returns branch length metrics (all branches taken into account and external only)
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param target_avg_length: float, the average branch length to which we want to rescale the tree
    :return: float, resc_factor
    """
    # branch lengths
    dist_all = [node.dist for node in tre.traverse("levelorder")]

    all_bl_mean = np.mean(dist_all)

    resc_factor = all_bl_mean/target_avg_length

    for node in tre.traverse():
        node.dist = node.dist/resc_factor

    return resc_factor

    


def encode_into_most_recent(tree_input, max_taxa=[500], summ_stat=[], target_average_brlen=1.0):
    """Rescales all trees from tree_file so that mean branch length is 1,
    then encodes them into full tree representation (most recent version)

    :param tree_input: ete3.Tree, that we will represent in the form of a vector
    :param sampling_proba: float, value between 0 and 1, presumed sampling probability value
    :return: pd.Dataframe, encoded rescaled input trees in the form of most recent, last column being
     the rescale factor
    """
    leaf_ordered_names = [] # A.M.T
    new_leaf_order_names = []
    newLeafKeys_inputNameValues = {}

    # do we want nested functions like this???
    def real_polytomies(tre):
        """
        Replaces internal nodes of zero length with real polytomies.
        :param tre: ete3.Tree, the tree to be modified
        :return: void, modifies the original tree
        """
        for nod in tre.traverse("postorder"):
            if not nod.is_leaf() and not nod.is_root():
                if nod.dist == 0:
                    for child in nod.children:
                        nod.up.add_child(child)
                    nod.up.remove_child(nod)
        return

    def get_not_visited_anc(leaf):
        while getattr(leaf, "visited", 0) >= len(leaf.children)-1:
            leaf = leaf.up
            if leaf is None:
                break
        return leaf

    def get_deepest_not_visited_tip(anc):
        max_dist = -1
        tip = None
        for leaf in anc:
            if leaf.visited == 0:
                distance_leaf = getattr(leaf, "dist_to_root") - getattr(anc, "dist_to_root")
                if distance_leaf > max_dist:
                    max_dist = distance_leaf
                    tip = leaf
        leaf_ordered_names.append(getattr(tip, "name")) # A.M.T
        return tip

    def get_dist_to_root(anc):
        dist_to_root = getattr(anc, "dist_to_root")
        return dist_to_root

    def get_dist_to_anc(feuille, anc):
        dist_to_anc = getattr(feuille, "dist_to_root") - getattr(anc, "dist_to_root")
        return dist_to_anc

    def encode(anc):
        leaf = get_deepest_not_visited_tip(anc)
        new_leaf_order_names.append(leaf.name) # A.M.T.
        yield get_dist_to_anc(leaf, anc)
        leaf.visited += 1
        anc = get_not_visited_anc(leaf)

        if anc is None:
            return
        anc.visited += 1
        yield get_dist_to_root(anc)
        for _ in encode(anc):
            yield _

    def complete_coding(encoding, cblv_length):
        #print(encoding, max_length, max_length - len(encoding) )
        add_vect = np.repeat(0, cblv_length - len(encoding))
        add_vect = list(add_vect)
        encoding.extend(add_vect)
        return encoding

    def refactor_to_final_shape(result_v, maxl, summ_stat=[]):
        def reshape_coor(max_length):
            tips_coor = np.arange(0, max_length, 2)  # second row
            #tips_coor = np.insert(tips_coor, -1, max_length + 1)
            int_nodes_coor = np.arange(1, max_length - 1, 2) # first row
            int_nodes_coor = np.insert(int_nodes_coor, 0, max_length) # prepend 0??
            #int_nodes_coor = np.insert(int_nodes_coor, -1, max_length + 2)
            order_coor = np.append(int_nodes_coor, tips_coor)
            return order_coor
       
        #print('test')
        reshape_coordinates = reshape_coor(maxl)

        #print(reshape_coordinates.shape)
        result_v.loc[:, maxl] = 0

        # append summ stats to final columns
        
        # # add sampling probability:
        # if maxl == 999:
        #     result_v.loc[:, 1000] = 0
        #     result_v['1001'] = sampling_p
        #     result_v['1002'] = sampling_p
        # else:
        #     result_v.loc[:, 400] = 0
        #     result_v['401'] = sampling_p
        #     result_v['402'] = sampling_p

        # reorder the columns        
        result_v = result_v.iloc[:,reshape_coordinates]

        return result_v

    # local copy of input tree
    tree = tree_input.copy()
    
    #if len(tree) < 200:
    #    max_len = 399
    num_summ_stat = len(summ_stat)

    cblv_length = 2*(max_taxa + num_summ_stat)
    # cblv_length = -1
    # for mt in max_taxa:
    #     if len(tree) <= mt:
    #         cblv_length = 2*(mt + num_summ_stat)
    #         break

    # if cblv_length == -1:
    #     raise Exception('tree too large')

    # remove the edge above root if there is one
    if len(tree.children) < 2:
        tree = tree.children[0]
        tree.detach()

    # set to real polytomy
    real_polytomies(tree)

    # rescale branch lengths
    rescale_factor = rescale_tree(tree, target_avg_length=target_average_brlen)

    # set all nodes to non visited:
    for node in tree.traverse():
        setattr(node, "visited", 0)

    name_tree(tree, newLeafKeys_inputNameValues)
    
    add_dist_to_root(tree)

    tree_embedding = list(encode(tree))
    
    tree_embedding = complete_coding(tree_embedding, cblv_length)
   
    result = pd.DataFrame(tree_embedding, columns=[0])

    result = result.T
 
    result = refactor_to_final_shape(result, cblv_length)

    return result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues

def read_tree(newick_tree):
    """ Tries all nwk formats and returns an ete3 Tree

    :param newick_tree: str, a tree in newick format
    :return: ete3.Tree
    """
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(newick_tree, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(newick_tree))
    return tree


def read_tree_file(tree_path):
    with open(tree_path, 'r') as f:
        nwk = f.read().replace('\n', '').split(';')
        if nwk[-1] == '':
            nwk = nwk[:-1]
    if not nwk:
        raise ValueError('Could not find any trees (in newick format) in the file {}.'.format(tree_path))
    if len(nwk) > 1:
        raise ValueError('There are more than 1 tree in the file {}. Now, we accept only one tree per inference.'.format(tree_path))
    return read_tree(nwk[0] + ';')


def check_tree_size(tre):
    """
    Verifies whether input tree is of correct size and determines the tree size range for models to use
    :param tre: ete3.Tree
    :return: int, tree_size
    """
    if 49 < len(tre) < 200:
        tre_size = 'SMALL'
    elif 199 < len(tre) < 501:
        tre_size = 'LARGE'
    else:
        raise ValueError('Your input tree is of incorrect size (either smaller than 50 tips or larger than 500 tips.')

    return tre_size


def annotator(predict, mod):
    """
    annotates the pd.DataFrame containing predicted values
    :param predict: predicted values
    :type: pd.DataFrame
    :param mod: model under which the parameters were estimated
    :type: str
    :return:
    """

    if mod == "BD":
        predict.columns = ["R_naught", "Infectious_period"]
    elif mod == "BDEI":
        predict.columns = ["R_naught", "Infectious_period", "Incubation_period"]
    elif mod == "BDSS":
        predict.columns = ["R_naught", "Infectious_period", "X_transmission", "Superspreading_individuals_fraction"]
    elif mod == "BD_vs_BDEI_vs_BDSS":
        predict.columns = ["Probability_BDEI", "Probability_BD", "Probability_BDSS"]
    elif mod == "BD_vs_BDEI":
        predict.columns = ["Probability_BD", "Probability_BDEI"]
    return predict


def rescaler(predict, rescale_f):
    """
    rescales the predictions back to the initial tree scale (e.g. days, weeks, years)
    :param predict: predicted values
    :type: pd.DataFrame
    :param rescale_f: rescale factor by which the initial tree was scaled
    :type: float
    :return:
    """

    for elt in predict.columns:
        if "period" in elt:
            predict[elt] = predict[elt]*rescale_f

    return predict


# helper functions
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def find_taxon_size(num_taxa, max_taxa):
    if num_taxa == 0:
        return 0
    elif num_taxa > max_taxa[-1]:
        return -1
    for i in max_taxa:
        if num_taxa <= i:
            return i
    # should never call this
    raise Exception('error in find_taxon_size()', num_taxa, max_taxa)
    return -2
    
# generate event map
def make_events(regions, states, states_inv):

    # make data structure
    num_regions = len(regions)
    events = { 'extinction': [],
               'extirpation': [],
               'dispersal': [],
               'within_region_speciation': [],
               'between_region_speciation': [] }

    # populate events
    for state,region_set in enumerate(states):
        if len(region_set) == 1:
            # extinction
            events['extinction'].append([state,region_set[0]])
        elif len(region_set) > 1:
            # between-region speciation
            splits = list(powerset(region_set))[1:-1]
            for x in splits:
                y = tuple(set(region_set).difference(x))
                x_state = states_inv[x]
                y_state = states_inv[y]
                z = [x_state, y_state]
                z.sort()
                v = [state, z[0], z[1]]
                if v not in events['between_region_speciation']:
                    events['between_region_speciation'].append( [state, z[0], z[1]] )
        for region in region_set:
            if len(region_set) > 1:
                # extirpation
                new_regions = tuple( set(region_set).difference([region]) )
                new_state = states_inv[new_regions]
                #print(state,region,new_regions,new_state)
                events['extirpation'].append( [state, new_state, region ] )
            # within-region speciation
            w_new_state = states_inv[ tuple([region]) ]
            events['within_region_speciation'].append( [state, w_new_state, region] )
            for dest in regions:
                if dest not in region_set:
                    i = tuple(set(region_set).union([dest]))
                    d_new_state = states_inv[i] 
                    v = [state, d_new_state, dest]
                    if v not in events['dispersal']:
                        events['dispersal'].append( [state, d_new_state, dest] )

    return events

def make_rates(regions, states, events, settings):

    # get settings
    model_type       = settings['model_type']
    num_regions      = len(events['extinction'])
    num_feature_layers = settings['num_feature_layers']
    rv_rate          = settings['rv_rate']
    rv_effect        = settings['rv_effect']
    rv_feature       = settings['rv_feature']

    # model rates
    if model_type == 'iid':
        r_e = rv_rate(size=num_regions)
        r_w = rv_rate(size=num_regions)
        r_d = rv_rate(size=(num_regions,num_regions))
        r_b = rv_rate(size=(num_regions,num_regions))

    elif model_type == 'iid_simple':
        r_e = np.repeat( rv_rate(size=1), num_regions )
        r_w = np.repeat( rv_rate(size=1), num_regions )
        r_d = np.repeat( rv_rate(size=1), num_regions*num_regions ).reshape(num_regions,num_regions)
        r_b = np.repeat( rv_rate(size=1), num_regions*num_regions ).reshape(num_regions,num_regions)

    elif model_type == 'FIG':
        # generate base rates
        rho_e, rho_w, rho_d, rho_w = rv_rate(size=4)

        # generate features and effects
        features = { 'cw':[], 'cb':[], 'qw':[], 'qb':[] }
        phi_effects = {'d':[], 'e':[], 'w':[], 'b':[] }
        sigma_effects = {'d':[], 'e':[], 'w':[], 'b':[] }

        for k in range(num_feature_layers):
            # regional features
            features['cw'].append( sp.stats.bernoulli.rvs(p=0.5, size=num_regions) )
            features['cb'].append( sp.stats.bernoulli.rvs(p=0.5, size=(num_regions,num_regions)) )
            features['qw'].append( rv_feature(size=num_regions) )
            features['qb'].append( rv_feature(size=(num_regions,num_regions)) )

            # feature effect parameters
            phi_effects['d'], phi_effects['e'], phi_effects['w'], phi_effects['b'] = rv_effect(size=4)
            sigma_effects['d'], sigma_effects['e'], sigma_effects['w'], sigma_effects['b'] = rv_effect(size=4)

        # normalize features

        # generate m relative rates

        # split scores

        # get absolute rates

    # collect rates
    rates = { 'r_e':r_e, 'r_w':r_w, 'r_b':r_b, 'r_d':r_d }

    # return rates
    return rates

# generate event rates xml
def make_xml(events, rates, states, states_str, settings):

    # XML settings
    max_taxa    = settings['max_taxa']
    out_path    = settings['out_path']
    newick_fn   = out_path + '.tre'
    nexus_fn    = out_path + '.nex'
    json_fn     = out_path + '.json'
    num_regions = len(events['extinction'])

    # rates
    r_e = rates['r_e']
    r_d = rates['r_d']
    r_w = rates['r_w']
    r_b = rates['r_b']

    # NOTE: uniform root state sampling is not ideal
    start_state = random.sample(states_str, 1)[0]

    # states
    xml_states = "<populationType spec='PopulationType' typeName='X' id='X'/>\n"
    for st in states_str:
        xml_states += "<populationType spec='PopulationType' typeName='{st}' id='{st}'/>\n".format(st=st)

    # extinction
    xml_extinction = "<reactionGroup spec='ReactionGroup' reactionGroupName='Extinction'>\n"
    for k,v in enumerate(events['extinction']):
        xml_extinction += "\t<reaction spec='Reaction' reactionName='Extinction-{i}' rate='{r}'>\n\t\t{i}:1 -> X\n\t</reaction>\n".format(i=states_str[v[0]], r=r_e[v[1]])
    xml_extinction += "</reactionGroup>\n"

    # extirpation
    xml_extirpation = "<reactionGroup spec='ReactionGroup' reactionGroupName='Extirpation'>\n"
    for k,v in enumerate(events['extirpation']):
        xml_extirpation += "\t<reaction spec='Reaction' reactionName='Extirpation-{i},{j}' rate='{r}'>\n\t\t{i}:1 -> {j}:1\n\t</reaction>\n".format(i=states_str[v[0]], j=states_str[v[1]], r=r_e[v[2]])
    xml_extirpation += "</reactionGroup>\n"

    # within-region speciation
    xml_within_speciation = "<reactionGroup spec='ReactionGroup' reactionGroupName='WithinRegionSpeciation'>\n"
    for k,v in enumerate(events['within_region_speciation']):
        xml_within_speciation += "\t<reaction spec='Reaction' reactionName='WithinRegionSpeciation-{i},{i},{j}' rate='{r}'>\n\t\t{i}:1 -> {i}:1 + {j}:1\n\t</reaction>\n".format(i=states_str[v[0]], j=states_str[v[1]], r=r_w[v[2]])
    xml_within_speciation += "</reactionGroup>\n"

    # dispersal
    xml_dispersal = "<reactionGroup spec='ReactionGroup' reactionGroupName='Dispersal'>\n"
    for k,v in enumerate(events['dispersal']):
        tmp_rate = 0.0
        tmp_range = states[ v[0] ]
        j = v[2]
        for i in tmp_range:
            tmp_rate += r_d[i][j]
        xml_dispersal += "\t<reaction spec='Reaction' reactionName='Dispersal-{i},{j}' rate='{r}'>\n\t\t{i}:1 -> {j}:1\n\t</reaction>\n".format(i=states_str[v[0]], j=states_str[v[1]], r=tmp_rate)
    xml_dispersal += "</reactionGroup>\n"

    # between-region speciation
    xml_between_speciation = "<reactionGroup spec='ReactionGroup' reactionGroupName='BetweenRegionSpeciation'>\n"
    for k,v in enumerate(events['between_region_speciation']):
        xml_between_speciation += "\t<reaction spec='Reaction' reactionName='BetweenRegionSpeciation-{i},{j},{k}' rate='{r}'>\n\t\t{i}:1 -> {j}:1 + {k}:1\n\t</reaction>\n".format(i=states_str[v[0]], j=states_str[v[1]], k=states_str[v[2]], r=1.0)
    xml_between_speciation += "</reactionGroup>\n"

    # collect XML model settings
    xml_rates_str = "\n".join([xml_states, xml_extinction, xml_extirpation, xml_within_speciation, xml_dispersal, xml_between_speciation ])

    # generate entire XML specification
    xml_spec_str = '''
<beast version='2.0' namespace='master:master.model:master.steppers:master.conditions:master.postprocessors:master.outputs'>

<run spec='InheritanceEnsemble'
    verbosity='3'
    nTraj='1'
    nSamples='{num_samples}'
    samplePopulationSizes='{sample_pop}'
    simulationTime='10'
    maxConditionRejects='1'>

<model spec='Model'>

{xml_rates}

</model>

<initialState spec='InitState'>
    <lineageSeedMultiple spec='MultipleIndividuals' copies='1' >
            <population spec='Population' type='@{start_state}'/>
    </lineageSeedMultiple>
</initialState>

<lineageEndCondition spec='LineageEndCondition' nLineages='{max_taxa}'
    alsoGreaterThan='true' isRejection='false'/>

<lineageEndCondition spec='LineageEndCondition' nLineages='0'
    alsoGreaterThan='false' isRejection='false'/>

<postSimCondition spec='LeafCountPostSimCondition' nLeaves='10'
    exact='false' exceedCondition='true'/>

<output spec='NewickOutput' collapseSingleChildNodes='true' fileName='{newick_fn}'/>
<output spec='NexusOutput' fileName='{nexus_fn}'/>
<output spec='JsonOutput' fileName='{json_fn}' />

</run>
</beast>
    '''.format(xml_rates=xml_rates_str, start_state=start_state, newick_fn=newick_fn, nexus_fn=nexus_fn, json_fn=json_fn, max_taxa=max_taxa, num_samples=1, sample_pop='false')


    return xml_spec_str


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
                s1 += '{k},{i},{i},{v}\n'.format(k=k,i=i,v=x)
                s2 += '{k}_{i},'.format(k=k,i=i)
                s3 += str(x) + ','
            else:
                for j,y in enumerate(x):
                    s1 += '{k},{i},{j},{v}\n'.format(k=k,i=i,j=j,v=y)
                    s2 += '{k}_{i}_{j},'.format(k=k,i=i,j=j)
                    s3 += str(y) + ','

    s4 = s2.rstrip(',') + '\n' + s3.rstrip(',') + '\n'
    return s1,s4


def regions_to_binary(states, states_str, regions):
    num_regions = len(regions)
    x = {}
    for i,v in enumerate(states):
        x[ states_str[i] ] = ['0']*num_regions
        for j in v:
            x[states_str[i]][j] = '1'
    return x

def prune_phy(tre_fn, prune_fn):
    # read tree
    tre_file = open(tre_fn, 'r')
    # prune non-extant taxa
    # write pruned tree
    return

def categorize_sizes(raw_data_dir):
    # get all files
    # learn sizes from param files
    # sort into groups
    # return dictionary of { size_key: [ index_list ] }
    return

def convert_geo_nex(nex_fn, tre_fn, geo_fn, states_bits):

    # get num regions from size of bit vector
    num_regions = len(list(states_bits.values())[0])

    # get tip names and states
    nex_file = open(nex_fn, 'r')
    nex_str = nex_file.readlines()[3]
    m = re.findall(pattern='([0-9]+)\[\&type="([A-Z]+)"', string=nex_str)
    num_taxa = len(m)
    nex_file.close()

    # generate taxon-state data
    d = {}
    s_state_str = ''
    for i,v in enumerate(m):
        taxon = v[0]
        state = ''.join(states_bits[v[1]])
        s_state_str += taxon + '  ' + state + '\n'
        d[ taxon ] = state
    
    # get newick string
    tre_file = open(tre_fn, 'r')
    #print(tre_file)
    tre_str = tre_file.readlines()[0]
    tre_file.close()

    # build new geosse string

    s = \
'''#NEXUS
Begin DATA;
Dimensions NTAX={num_taxa} NCHAR={num_regions}
Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
Matrix
{s_state_str}
;
END;

Begin trees;
    tree 1={tre_str}
END;
'''.format(num_taxa=num_taxa, num_regions=num_regions, tre_str=tre_str, s_state_str=s_state_str)


    geo_file = open(geo_fn, 'w')
    geo_file.write(s)
    geo_file.close()

    return d

def vectorize_tree_cdv(tre_fn, max_taxa=[500], summ_stat=[], prob=1.0):
    # get tree and tip labels
    tree = read_tree_file(tre_fn)    
    ordered_tip_names = []
    for i in tree.get_leaves():
        ordered_tip_names.append(i.name)

    # returns result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues
    vv = encode_into_most_recent(tree, max_taxa=max_taxa, summ_stat=summ_stat, target_average_brlen=1.0)
    otn = np.asarray(ordered_tip_names) # ordered list of the input tip labels
    vv2 = np.asarray(vv[2]) # ordered list of the new tip labels
    new_order = [vv[3][i] for i in vv2]

    if False:
        print( 'otn ==> ', otn, '\n' )
        print( 'vv[0] ==>', vv[0], '\n' )
        print( 'vv[1] ==>', vv[1], '\n' )
        print( 'vv[2] ==>', vv[2], '\n' )
        print( 'vv[3] ==>', vv[3], '\n' )

    cblv = np.asarray( vv[0] )
    cblv.shape = (2, -1)
    cblv_df = pd.DataFrame( cblv )

    return cblv_df,new_order

def vectorize_tree(tre_fn, max_taxa=500, summ_stat=[], prob=1.0):

    # get tree and tip labels
    tree = read_tree_file(tre_fn)    
    ordered_tip_names = []
    for i in tree.get_leaves():
        ordered_tip_names.append(i.name)

    # returns result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues
    vv = encode_into_most_recent(tree, max_taxa=max_taxa, summ_stat=summ_stat, target_average_brlen=1.0)
    otn = np.asarray(ordered_tip_names) # ordered list of the input tip labels
    vv2 = np.asarray(vv[2]) # ordered list of the new tip labels
    new_order = [vv[3][i] for i in vv2]

    if False:
        print( 'otn ==> ', otn, '\n' )
        print( 'vv[0] ==>', vv[0], '\n' )
        print( 'vv[1] ==>', vv[1], '\n' )
        print( 'vv[2] ==>', vv[2], '\n' )
        print( 'vv[3] ==>', vv[3], '\n' )

    cblv = np.asarray( vv[0] )
    cblv.shape = (2, -1)
    cblv_df = pd.DataFrame( cblv )

    return cblv_df,new_order


def make_cblvs_geosse(cblv_df, taxon_states, new_order):
    
    # array dimensions for GeoSSE states
    n_taxon_cols = cblv_df.shape[1]
    n_region = len(list(taxon_states.values())[0])

    # create states array
    states_df = np.zeros( shape=(n_region,n_taxon_cols), dtype='int')
    
    # populate states (not sure if this is how new_order works!)
    for i,v in enumerate(new_order):
        y =  [ int(x) for x in taxon_states[v] ]
        z = np.array(y, dtype='int' )
        states_df[:,i] = z

    # append states
    cblvs_df = np.concatenate( (cblv_df, states_df), axis=0 )
    cblvs_df = cblvs_df.T.reshape((1,-1))
    #cblvs_df.shape = (1,-1)

    # done!
    return cblvs_df

def make_cdv_geosse(cdv_df, taxon_states, new_order):
    
    # array dimensions for GeoSSE states
    n_taxon_cols = cblv_df.shape[1]
    n_region = len(list(taxon_states.values())[0])

    # create states array
    states_df = np.zeros( shape=(n_region,n_taxon_cols), dtype='int')
    
    # populate states (not sure if this is how new_order works!)
    for i,v in enumerate(new_order):
        y =  [ int(x) for x in taxon_states[v] ]
        z = np.array(y, dtype='int' )
        states_df[:,i] = z

    # append states
    cblvs_df = np.concatenate( (cblv_df, states_df), axis=0 )
    cblvs_df = cblvs_df.T.reshape((1,-1))
    #cblvs_df.shape = (1,-1)

    # done!
    return cblvs_df

def write_to_file(s, fn):
    f = open(fn, 'w')
    f.write(s)
    f.close()

def get_num_taxa(tre_fn, k, max_taxa):
    tree = None
    try:
        tree = read_tree_file(tre_fn)
        n_taxa_k = len(tree.get_leaves())
        #if n_taxa_k > np.max(max_taxa):
        #    #warning_str = '- replicate {k} simulated n_taxa={nt} (max_taxa={mt})'.format(k=k,nt=n_taxa_k,mt=max_taxa)
        #    #print(warning_str)
        #    return n_taxa_k #np.max(max_taxa)
    except ValueError:
        #warning_str = '- replicate {k} simulated n_taxa=0'.format(k=k)
        #print(warning_str)
        return 0
    
    return n_taxa_k


def init_sim_settings(settings):
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle settings')
    parser.add_argument('--name', dest='model_name', type=str, help='Model name')
    parser.add_argument('--start_idx', dest='start_idx', type=int, help='Start index for range of replicates')
    parser.add_argument('--end_idx', dest='end_idx', type=int, help='End index for range of replicates')
    parser.add_argument('--cfg', dest='cfg', type=str, help='Model configuration file')
    parser.add_argument('--use_parallel', action=argparse.BooleanOptionalAction, help='Use parallelization?')
    args = parser.parse_args()

    print(args)
    # set arguments
    if args.model_name != None:
        settings['model_name'] = args.model_name
    if args.start_idx != None:
        settings['start_idx'] = args.start_idx
    if args.end_idx != None:
        settings['end_idx'] = args.end_idx
    if args.cfg != None:
        settings['cfg_fn'] = args.cfg
    if args.use_parallel != None:
        settings['use_parallel'] = args.use_parallel

    return settings

def init_process_settings(settings):
    parser = argparse.ArgumentParser('data process settings')
    parser.add_argument('--name', dest='model_name', type=str, help='Model name')
    args = parser.parse_args()
    if args.model_name != None:
        settings['model_name'] = args.model_name
    return settings

def init_cnn_settings(settings):
    parser = argparse.ArgumentParser('CNN settings')
    parser.add_argument('--name', dest='model_name', type=str, help='Model name')
    parser.add_argument('--prefix', dest='prefix', type=str, help='Simulation prefix')
    parser.add_argument('--num_epoch', dest='num_epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size')
    parser.add_argument('--num_validation', dest='num_validation', type=int, help='Number of validation records')
    parser.add_argument('--num_test', dest='num_test', type=int, help='Number of test records')
    parser.add_argument('--max_taxa', dest='max_taxa', type=int, help='Number of taxon slots in CBLV record')
    args = parser.parse_args()
    if args.model_name != None:
        settings['model_name'] = args.model_name
    if args.prefix != None:
        settings['prefix'] = args.prefix
    if args.num_epoch != None:
        settings['num_epoch'] = args.num_epoch
    if args.batch_size != None:
        settings['batch_size'] = args.batch_size
    if args.num_validation != None:
        settings['num_validation'] = args.num_validation
    if args.num_test != None:
        settings['num_test'] = args.num_test
    if args.max_taxa != None:
        settings['max_taxa'] = args.max_taxa
    return settings

class BatchCompletionCallBack(object):
    # Added code - start
    global total_n_jobs
    # Added code - end
    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel

    def __call__(self, out):
        self.parallel.n_completed_tasks += self.batch_size
        this_batch_duration = time.time() - self.dispatch_timestamp

        self.parallel._backend.batch_completed(self.batch_size,
                                           this_batch_duration)
        self.parallel.print_progress()
        # Added code - start
        progress = self.parallel.n_completed_tasks / total_n_jobs
        print(
            "\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress*100)
            , end="", flush=True)
        if self.parallel.n_completed_tasks == total_n_jobs:
            print('\n')
        # Added code - end
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()


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
