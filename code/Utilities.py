# general libraries
import argparse
import importlib
import sys
import random
import re
import os

# Call before importing Tensorflow to suppress INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import pandas as pd
import numpy as np
import scipy as sp
import dendropy as dp
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import metrics
from collections import Counter
from itertools import chain, combinations
from ete3 import Tree

NUM_DIGITS = 10
np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
pd.set_option('display.precision', NUM_DIGITS)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

#max_len = 501
TURN_ONE = 'turn_one'

# the information on state is saved as 't_s' in the newick tree
T_S = 't_s'
STATE = 'state'
DIVERSIFICATION_SCORE = 'diversification_score'

sys.setrecursionlimit(100000)

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



#################
# FILE HANDLERS #
#################

def load_config(config_fn, arg_overwrite=True):
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config')
    parser.add_argument('--cfg', dest='config_fn', type=str, help='Config file name')
    parser.add_argument('--job', dest='job_name', type=str, help='Job directory')
    parser.add_argument('--start_idx', dest='start_idx', type=int, help='Start index for simulation')
    parser.add_argument('--end_idx', dest='end_idx', type=int, help='End index for simulation')
    args = parser.parse_args()
    
    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn != None:
        config_fn = args.config_fn
    
    # config from file
    m = importlib.import_module(config_fn)

    # overwrite default args
    if args.job_name != None:
        m.my_all_args['job_name'] = args.job_name
    if args.start_idx != None:
        m.my_sim_args['start_idx'] = args.start_idx
    if args.end_idx != None:
        m.my_sim_args['end_idx'] = args.end_idx

    # update args    
    m.my_sim_args = m.my_sim_args | m.my_all_args
    m.my_fmt_args = m.my_fmt_args | m.my_all_args
    m.my_lrn_args = m.my_lrn_args | m.my_all_args

    # return new args
    return m



#################
# MODEL HELPERS #
#################

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


################
# CDVS ENCODER #
################

### this is where state info is stored
def set_attribs(tre):
    """
    adds t_s attributes to tips based on tip name
    :param tre: ete3.Tree, the tree on which we measure the branch length
    :return: void, returns modified tree
    """
    for tip in tre.traverse():
        if "&&NHX-t_s=1" in tip.name:
            setattr(tip, T_S, 1)
        elif "&&NHX-t_s=2" in tip.name:
            setattr(tip, T_S, 2)
    return None

def attach_tip_states(tr, st):
    "assign states in st to leaf nodes in tr using shared taxon name"
    for nd in tr.get_leaves():
        setattr(nd, STATE, st[nd.name])

def get_average_branch_length(tre):
    """
    Returns average branch length for given tree
    :param tre: ete3.Tree, the tree on which we measure the branch length
    :return: float, average branch length
    """
    br_length = [nod.dist for nod in tre.traverse()]
    return np.average(br_length)


def rescale_tree(tr, rescale_fac):
    """
    Rescales a given tree
    :param tr: ete3.Tree, the tree to be rescaled
    :param rescale_fac: float, the branches will be multiplied by this factor
    :return: void, modifies the original tree
    """
    for node in tr.traverse():
        node.dist = node.dist/rescale_fac
    return None


def add_diversification(tr):
    """
    to each node adds an attribute, 'diversification_score', i.e. the sum of pathways of branched tips
    :param tr: ete3.Tree, the tree to be modified
    :return: void, modifies the original tree
    """
    for node in tr.traverse("postorder"):
        if not node.is_root():
            # print(label_count)
            label_node = 0
            if node.is_leaf():
                label_node = 1
                setattr(node, DIVERSIFICATION_SCORE, node.dist)
            else:
                children = node.get_children()
                # print(children)
                setattr(node, DIVERSIFICATION_SCORE, getattr(children[0], DIVERSIFICATION_SCORE) + getattr(children[1], DIVERSIFICATION_SCORE))
    return None


def add_diversification_sign(tr):
    """
    Puts topological signatures based on diversification (i.e. longest path): if the first child of a node has longer
    path of branches leading to it, then it is prioritized for visit.
    :param tr: ete3.Tree, the tree to get the topological description
    :return: void, modifies the original tree
    """
    for node in tr.traverse('levelorder'):
        if not node.is_leaf():
            diver_child0 = getattr(node.children[0], DIVERSIFICATION_SCORE)
            diver_child1 = getattr(node.children[1], DIVERSIFICATION_SCORE)
            if diver_child0 < diver_child1:
                node.add_feature(TURN_ONE, True)
            elif diver_child0 == diver_child1:
                next_sign = random.choice([True, False])
                if next_sign is True:
                    node.add_feature(TURN_ONE, True)
            else:
                node.add_feature(TURN_ONE, False)
    return None


def name_tree_cdvs(tr):
    """
    Names all the tree nodes that are not named, with unique names.
    :param tr: ete3.Tree, the tree to be named
    :return: void, modifies the original tree
    """
    i = 0
    for node in tr.traverse('levelorder'):
        node.name = i
        i += 1
    return None


def add_dist_to_root_cdvs(tr):
    # int_nodes_dist = []
    # tips_dist = []
    tree_height = 0
    for node in tr.traverse("preorder"):
        if node.is_root():
            node.add_feature("dist_to_root", 0)
        elif node.is_leaf():
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
            tree_height = getattr(node, "dist_to_root", False)

        else:
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
    return tr, tree_height


def get_not_visited_anc(leaf):
    while getattr(leaf, "visited", False):
        leaf = leaf.up
    return leaf


def get_dist_to_root(anc):
    dist_to_root = getattr(anc, "dist_to_root")
    return dist_to_root


def follow_signs(anc):
    end_leaf = anc
    while not end_leaf.is_leaf():
        if getattr(end_leaf, TURN_ONE, False):
            if getattr(end_leaf.children[1], 'visited', False):
                end_leaf = end_leaf.children[0]
            else:
                end_leaf = end_leaf.children[1]
        else:
            if getattr(end_leaf.children[0], 'visited', False):
                end_leaf = end_leaf.children[1]
            else:
                end_leaf = end_leaf.children[0]
    return end_leaf


def enc_diver(anc):
    leaf = follow_signs(anc)
    #print(leaf.state)
    #print([ int(x) for x in leaf.state] )
    #s = [ int(x) for x in leaf.state ]
    yield str(leaf.state)
    #for s in [ int(x) for x in leaf.state ]:
    #    yield s
    #yield float(leaf.state)
    setattr(leaf, 'visited', True)
    anc = get_not_visited_anc(leaf)
    if anc is None:
        # print("what")
        return
    setattr(anc, 'visited', True)
    yield get_dist_to_root(anc)
    for _ in enc_diver(anc):
        yield _


def type_count(tr, st, lbl):
    """
    Returns the counts of type1 and type2 tips
    :param tr: ete3.Tree, the tree to be named
    :return: tuple, counts of type 1 and type 2
    """
    counts = dict.fromkeys(lbl, 0)
    #t1 = 0
    #t2 = 0
    for leaf in tr:
        counts[ leaf.state ] = counts[ leaf.state ] + 1
    
    #print(counts)
    return list(counts.values())


def complete_coding_old(encoding, max_length):
    add_vect = np.repeat(0, max_length - len(encoding))
    add_vect = list(add_vect)
    encoding.extend(add_vect)
    return encoding

def complete_coding(encoding, max_length):
    #add_vect = np.repeat(0, max_length - len(encoding))
    num_row,num_col = encoding.shape
    add_zeros = np.zeros( (num_row, max_length-num_col) )
    #add_vect = np.repeat(0, max_length - num_col)
    #encoding = np.append(encoding, add_vect)
    #encoding.extend(add_vect)
    encoding = np.hstack( [encoding, add_zeros] )
    return encoding

def expand_tip_states(tips_info):
    n_idx = len(tips_info)
    n_char = len(tips_info[1])
    #print(n_char, n_idx)
    x = np.zeros( shape=(n_char, n_idx) )
    #x[:,0] = tips_info[0]  # needed?
    for i in range(n_idx):
        x[:,i] = [ int(y) for y in tips_info[i] ]
    return x


def make_cdvs(tree_fn, max_len, states, state_labels):

    file = open(tree_fn, mode="r")
    
    tree_str = file.read()
    
    tree = Tree(tree_str, format=1)

    attach_tip_states(tree, states)
    #set_attribs(tree)
    name_tree_cdvs(tree)

    #print('max_len ==>', max_len)
    #print('states ==>', states)

    # rescale tree to average branch length of 1
    # measure average branch length
    rescale_factor = get_average_branch_length(tree)

    # rescale tree
    rescale_tree(tree, rescale_factor)

    # add dist to root attribute
    tree, tr_height = add_dist_to_root_cdvs(tree)

    # add pathway of visiting priorities for encoding
    add_diversification(tree)
    add_diversification_sign(tree)

    # encode the tree
    tree_embedding = list(enc_diver(tree))
    
    # separate info on tips and nodes:
    tips_info = [tree_embedding[i] for i in range(len(tree_embedding)) if i % 2 == 0]
    node_info = [tree_embedding[i] for i in range(len(tree_embedding)) if i % 2 == 1]
    node_info.insert(0,0) # pad with zero to align length of info vec ??

    #print('tips_info ==> ', tips_info)
    #print('node_info ==>', node_info)

    # expand tip states
    tips_info = expand_tip_states(tips_info)
    node_info = np.array([node_info])

    # complete embedding
    tips_info = complete_coding(tips_info, max_len)
    node_info = complete_coding(node_info, max_len)

    # vertical stack
    #complete_info = np.vstack( [tips_info, node_info] )
    complete_info = np.vstack( [node_info, tips_info] )
    
    # extra info
    nrow = complete_info.shape[0]
    state_counts = type_count(tree, states, state_labels)
    extra_info = [ tr_height, rescale_factor ] + state_counts
    for x in extra_info:
        complete_info = np.append( complete_info, np.repeat(x, nrow).reshape(-1,1) )
    
    # flatten
    complete_info.reshape(-1)
    
    # make output
    result = pd.DataFrame(complete_info) #, columns=[id + 0])
    result = result.T

    return result



#################
# CBLVS ENCODER #
#################


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


##############
# READ TREES #
##############

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


# unused??
def regions_to_binary(states, states_str, regions):
    num_regions = len(regions)
    x = {}
    for i,v in enumerate(states):
        x[ states_str[i] ] = ['0']*num_regions
        for j in v:
            x[states_str[i]][j] = '1'
    return x

## set return None if bad, then flag the index as a bad sim.
def make_prune_phy(tre_fn, prune_fn):
    # read tree
    phy = dp.Tree.get(path=tre_fn, schema='newick')
    # compute all root-to-node distances
    root_distances = phy.calc_node_root_distances()
    # find tree height (max root-to-node distance)
    tree_height = np.max( root_distances )
    # create empty dictionary
    d = {}
    # loop through all leaf nodes
    leaf_nodes = phy.leaf_nodes()
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
    # abort if pruned tree would be invalid
    if len(leaf_nodes) - len(drop_taxon_labels) < 2:
        #print( "leaf_nodes ==>", leaf_nodes)
        #print( "drop_taxon_labels ==>", drop_taxon_labels )
        #print( "len(leaf_nodes) ==>", len(leaf_nodes))
        #print( "len(drop_taxon_labels) ==>", len(drop_taxon_labels) )
        return False
    else:
        # prune non-extant taxa
        phy.prune_taxa_with_labels( drop_taxon_labels )
        # write pruned tree
        phy.write(path=prune_fn, schema='newick')
        return True

# def categorize_sizes(raw_data_dir):
#     # get all files
#     # learn sizes from param files
#     # sort into groups
#     # return dictionary of { size_key: [ index_list ] }
#     return

def convert_nex(nex_fn, tre_fn, int2vec):

    # get num regions from size of bit vector
    num_char = len(int2vec[0])

    # get tip names and states from NHX tree
    nex_file = open(nex_fn, 'r')
    nex_str = nex_file.readlines()[3]
    m = re.findall(pattern='([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"', string=nex_str)
    num_taxa = len(m)
    nex_file.close()

    # generate taxon-state data
    d = {}
    s_state_str = ''
    for i,v in enumerate(m):
        taxon = v[0]
        state = int(v[2])
        vec_str = ''.join([ str(x) for x in int2vec[state] ])
        s_state_str += taxon + '  ' + vec_str + '\n'
        d[ taxon ] = vec_str
    
    # get newick string (no annotations)
    tre_file = open(tre_fn, 'r')
    tre_str = tre_file.readlines()[0]
    tre_file.close()

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

Begin trees;
    tree 1={tre_str}
END;
'''.format(num_taxa=num_taxa, num_char=num_char, tre_str=tre_str, s_state_str=s_state_str)

    return d,s

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

    #if False:
    #    print( 'otn ==> ', otn, '\n' )
    #    print( 'vv[0] ==>', vv[0], '\n' )
    #    print( 'vv[1] ==>', vv[1], '\n' )
    #    print( 'vv[2] ==>', vv[2], '\n' )
    #    print( 'vv[3] ==>', vv[3], '\n' )

    cblv = np.asarray( vv[0] )
    cblv.shape = (2, -1)
    cblv_df = pd.DataFrame( cblv )

    return cblv_df,new_order


#####################
# FORMAT CONVERSION #
#####################

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


#######################
# FILE/STRING HELPERS #
#######################

def write_to_file(s, fn):
    f = open(fn, 'w')
    f.write(s)
    f.close()


#####################################
# DATA DIMENSION/VALIDATION HELPERS #
#####################################

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

#################
# SUMMARY STATS #
#################

def make_summ_stat(tre_fn, geo_fn, states_bits_str_inv):
    
    # build summary stats
    summ_stats = {}

    # read tree + states
    phy = dp.Tree.get(path=tre_fn, schema="newick")
    num_taxa                  = len(phy.leaf_nodes())
    root_distances            = phy.calc_node_root_distances()
    tree_height               = np.max( root_distances )
    branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]

    # tree statistics
    summ_stats['n_taxa']      = num_taxa
    summ_stats['tree_length'] = phy.length()
    summ_stats['tree_height'] = tree_height
    summ_stats['brlen_mean']  = sp.stats.mean(branch_lengths)
    summ_stats['brlen_var']   = sp.stats.var(branch_lengths)
    summ_stats['brlen_skew']  = sp.stats.skew(branch_lengths)
    summ_stats['brlen_kurt']  = sp.stats.kurtosis(branch_lengths)
    summ_stats['age_mean']    = sp.stats.mean(root_distances)
    summ_stats['age_var']     = sp.stats.var(root_distances)
    summ_stats['age_skew']    = sp.stats.skew(root_distances)
    summ_stats['age_kurt']    = sp.stats.kurtosis(root_distances)
    summ_stats['B1']          = dp.calculate.treemeasure.B1(phy)
    summ_stats['N_bar']       = dp.calculate.treemeasure.N_bar(phy)
    summ_stats['colless']     = dp.calculate.treemeasure.colless_tree_imbalance(phy)
    summ_stats['treeness']    = dp.calculate.treemeasure.treeness(phy)
    #summ_stats['gamma']       = dp.calculate.treemeasure.pybus_harvey_gamma(phy)
    #summ_stats['sackin']      = dp.calculate.treemeasure.sackin_index(phy)

    # read characters + states
    f = open(geo_fn, 'r')
    m = f.read().splitlines()
    f.close()
    y = re.search(string=m[2], pattern='NCHAR=([0-9]+)')
    z = re.search(string=m[3], pattern='SYMBOLS="([0-9A-Za-z]+)"')
    num_char = int(y.group(1))
    states = z.group(1)
    #num_states = len(states)
    #num_combo = num_char * num_states

    # get taxon data
    taxon_state_block = m[ m.index('Matrix')+1 : m.index('END;')-1 ]
    taxon_states = [ x.split(' ')[-1] for x in taxon_state_block ]

    # freqs of entire char-set
    # freq_taxon_states = np.zeros(num_char, dtype='float')
    for i in range(num_char):
        summ_stats['n_char_' + str(i)] = 0
        summ_stats['f_char_' + str(i)] = 0.
    for k in list(states_bits_str_inv.keys()):
        #freq_taxon_states[ states_bits_str_inv[k] ] = taxon_states.count(k) / num_taxa
        summ_stats['n_state_' + str(k)] = taxon_states.count(k)
        summ_stats['f_state_' + str(k)] = taxon_states.count(k) / num_taxa
        for i,j in enumerate(k):
            if j != '0':
                summ_stats['n_char_' + str(i)] += summ_stats['n_state_' + k]
                summ_stats['f_char_' + str(i)] += summ_stats['f_state_' + k]

    return summ_stats

def make_summ_stat_str(ss):
    keys_str = ','.join( list(ss.keys()) ) + '\n'
    vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
    return keys_str + vals_str


# file handling
def load_input( data_fn, label_fn ):
    data = pd.read_csv(data_fn, header=None, on_bad_lines='skip').to_numpy()
    labels = pd.read_csv(label_fn, header=None, on_bad_lines='skip').to_numpy()
    return data,labels

# loss functions
def myLoss(y_true, y_pred):
    power = 2 # 3
    power_loss = tf.math.abs(y_true - y_pred)**power
    return tf.reduce_mean(power_loss, axis=-1)


def summarize_categorical_performance(y_true, y_pred):
    accuracy = np.max(y_pred * y_true[:,:5], axis = 1)
    auc = metrics.roc_auc_score(y_true[:,:5], y_pred)
    
    ### eps set due to 3 sig digits rounding in get_root_state_probs.sh script. 
    # set to midpoint between 0 and 0.001
    cross_entropy = metrics.log_loss(y_true[:,:5], y_pred, eps = 5e-4) 
    
    return accuracy, auc, cross_entropy

    
def tip_freq_accuracy(treeLocation_tensor, labels, num_locs = 5):

    tip_loc_counts = np.zeros((treeLocation_tensor.shape[0], num_locs))
    tip_loc_distro = np.zeros((treeLocation_tensor.shape[0], num_locs))
    accuracy_tipfreq = np.zeros((treeLocation_tensor.shape[0]))

    for i in range(0, treeLocation_tensor.shape[0]):    
        tip_loc_counts[i,:] = sum(treeLocation_tensor[i,:,2:2+num_locs])
        tip_loc_distro[i,:] = tip_loc_counts[i,:] / sum(tip_loc_counts[i,:])
        accuracy_tipfreq[i] = sum(tip_loc_distro[i,:] * labels[i,:5])
        
    return accuracy_tipfreq, tip_loc_distro

    
def get_num_tips(tree_data_tensor):
    # tree size
    num_sample = tree_data_tensor.shape[0]
    tree_data_tensor = tree_data_tensor.reshape((num_sample, 502, 7), order = 'C')
    num_tips = []
    for i in range(tree_data_tensor.shape[0]):
        num_tips.append(len(np.where(tree_data_tensor[i,:,0] > 0)[0]))
    num_tips = np.asarray(num_tips)
    
    return np.array(num_tips)


def normalize_01(data, min_max = None):
    if(type(min_max) == type(None)):
        max_value = data.max(axis = 0)
        min_value = data.min(axis = 0)
        difference = max_value - min_value
        difference[np.where(difference <= 0)] = 1
        return (max_value - data)/difference, min_value, max_value
    else:
        min_value = min_max[0]
        max_value = min_max[1]
        difference = max_value - min_value
        difference[np.where(difference <= 0)] = 1
        return (max_value - data)/difference

    
    
def normalize(data, m_sd = None):
    if(type(m_sd) == type(None )):
        m = data.mean(axis = 0)
        sd = data.std(axis = 0)
        sd[np.where(sd == 0)] = 1
        return (data - m)/sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0])/m_sd[1]
        
    

    
def denormalize(data, train_mean, train_sd, log_labels = False):
    return data * train_sd + train_mean


def denormalize_01(data, train_min, train_max):
    return train_max - data * (train_max - train_min)

        
def create_data_tensors2(data, mu, subsample_prop,
                            tmrca, mean_bl, num_tips, num_locs, max_tips,
                           cblv_contains_mu_rho = True):
    
    num_sample = data.shape[0]
    
    # reshape data tensor    
    full_data_tensor = data.reshape((num_sample, max_tips, num_locs + 2), order = 'C')

    # create tree/location tensor
    if(cblv_contains_mu_rho):
        full_treeLocation_tensor = full_data_tensor[:,:max_tips-3,:]
    else:
        full_treeLocation_tensor = full_data_tensor
        
    
    # create prior tensor
    subsample_prop = np.repeat(subsample_prop, 2)
    subsample_prop = subsample_prop.reshape((num_sample, 1, 2))
    mu = np.repeat(mu , 2)
    mu = mu.reshape((num_sample , 1, 2))
    num_tips = np.repeat(num_tips, 2)
    num_tips = num_tips.reshape((num_sample, 1, 2))
    tmrca = np.repeat(tmrca, 2)
    tmrca = tmrca.reshape((num_sample, 1, 2))
    mean_bl = np.repeat(mean_bl, 2)
    mean_bl = mean_bl.reshape((num_sample, 1, 2))
    
    full_prior_tensor = np.concatenate((mu, subsample_prop, num_tips, tmrca, mean_bl), axis = 1)
    
    return full_treeLocation_tensor, full_prior_tensor

def create_train_val_test_tensors(full_tensor, num_validation, num_test):
    # training tensors
    train_tensor = full_tensor[num_test + num_validation:,:,:]

    # validation tensors
    validation_tensor = full_tensor[num_test:num_test + num_validation,:,:]

    # testing tensors
    test_tensor = full_tensor[:num_test,:,:]

    return train_tensor, validation_tensor, test_tensor



#######################
# PLotting functions ##
#######################

def make_history_plot(history, prefix, plot_dir):
    epochs      = range(1, len(history.history['loss']) + 1)
    train_keys  = [ x for x in history.history.keys() if 'val' not in x ]
    val_keys = [ 'val_'+x for x in train_keys ]
    for i,v in enumerate(train_keys): #range(0,num_metrics):
        plt.plot(epochs, history.history[train_keys[i]], 'bo', label = train_keys[i])
        plt.plot(epochs, history.history[val_keys[i]], 'b', label = val_keys[i])
        plt.title('Train and val ' + train_keys[i])
        plt.xlabel('Epochs')
        plt.ylabel(train_keys[i])
        plt.legend()
        save_fn = plot_dir + '/' + prefix + '_' + train_keys[i] + '.pdf'
        plt.savefig(save_fn, format='pdf')
        plt.clf()


def plot_preds_labels(preds, labels, param_names, plot_dir, prefix, axis_labels = ["prediction", "truth"], title = ''):
    for i in range(0, len(param_names)):
        plt.title(title)
        plt.scatter(preds[:,i], labels[:,i], alpha =0.25)
        plt.xlabel(param_names[i] + " " +  axis_labels[0])
        plt.ylabel(param_names[i] + " " +  axis_labels[1])
        plt.axline((np.min(labels[:,i]),np.min(labels[:,i])), slope = 1, color = 'red', alpha=0.75)
        save_fn = plot_dir + '/' + prefix + '_' + param_names[i] + '.pdf'
        plt.savefig(save_fn, format='pdf')
        plt.clf()
        #plt.show()


def plot_root_pred_examples(labels, preds, phylo_post, tip_loc_distro, num_plots = 10, num_locs = 5):
    cats = np.arange(num_locs)
    barwidth = 0.2
    randidx = np.random.permutation(labels.shape[0]-1)[0:num_plots]
    for i in randidx:
        plt.figure(figsize=(num_locs+2, 1))
        plt.bar(cats + barwidth, preds[i,:], barwidth, label = "prediction")
        plt.bar(cats, labels[i,:5], barwidth, label = "truth")
        plt.bar(cats + 2 * barwidth, phylo_post[i,:], barwidth, label = "phylo", color = "red")
        plt.bar(cats + 3 * barwidth, tip_loc_distro[i,:], barwidth, label = 'tip frequency')
        plt.show()
    plt.close()


def root_summary_plots(cnn_root_accuracy, phylo_root_accuracy, accuracy_tipfreq):

    plt.hist(cnn_root_accuracy, bins = 20, range = [0,1], color = 'blue')
    plt.xlabel('CNN and Phylo accuracy')
    plt.hist(phylo_root_accuracy, bins = 20, range = [0,1], alpha = 0.5, color = 'red')
    plt.legend(['CNN', 'Phylo'])
    plt.show()

    plt.hist(phylo_root_accuracy - cnn_root_accuracy, bins = 20)
    plt.axline((0,0), slope = 100000, color = 'red', alpha=0.75)
    plt.xlabel("phylo_accuracy - cnn accuracy")
    plt.show()

    plt.hist(accuracy_tipfreq, bins = 20, range = [0,1])
    plt.xlabel('Tip frequency accuracy')
    plt.show()

    plt.scatter(cnn_root_accuracy, phylo_root_accuracy)
    plt.xlabel("CNN accuracy")
    plt.ylabel("phylo accuracy")
    plt.axline((np.min(cnn_root_accuracy),np.min(phylo_root_accuracy)), slope = 1, color = 'red', alpha=0.75)
    plt.show()


def plot_overlaid_scatter(sample_1, sample_2, reference_sample, 
                          sample_names = ['CNN', 'phylo'],
                          param_names = ["R0", "sample rate", "migration rate"], 
                          axis_labels = ["estimate", "truth"]):
    dot_colors = ['blue', 'red']
    for i in range(0, sample_1.shape[1]):
        minimum = np.min([sample_1[:,i], reference_sample[:,i]])
        plt.scatter(sample_1[:,i], reference_sample[:,i], alpha =0.75, color = dot_colors[0])
        plt.scatter(sample_2[:,i], reference_sample[:,i], alpha =0.75, color = dot_colors[1])
        plt.xlabel(param_names[i] + " " + axis_labels[0])
        plt.ylabel(param_names[i] + " " + axis_labels[1])
        plt.legend(sample_names)
        plt.axline((minimum, minimum), slope = 1, color = 'red', alpha=0.75)
        plt.show()
        

def plot_convlayer_weights(model, layer_num):
    layer_num = layer_num
    print(model.layers[layer_num].get_config())
    print(model.layers[layer_num].get_weights()[0].shape)
    layer_biases = model.layers[layer_num].get_weights()[1]
    layer_weights = model.layers[layer_num].get_weights()[0]
    for j in range(0, layer_weights.shape[2]):
        filter_num = j
        print(filter_num)
        for k in range(0,layer_weights.shape[1]):    
            plt.hlines(0,0,layer_weights.shape[0]-1, linestyle='dashed', color = "black")
            plt.plot(layer_weights[:,k,filter_num], color=np.random.rand(3,))
            plt.vlines(0,-0.5,0.5, color = "white")
        plt.show()
    
    
def plot_denselayer_weights(model, layer_num):
    layer_num = layer_num
    print(model.layers[layer_num].get_config())
    print(model.layers[layer_num].get_weights()[0].shape)
    layer_biases = model.layers[layer_num].get_weights()[1]
    layer_weights = model.layers[layer_num].get_weights()[0]
    for j in range(0, layer_weights.shape[1]):
        filter_num = j
        print(filter_num)
        plt.hlines(0,0,layer_weights.shape[0]-1, linestyle='dashed', color = "black")
        plt.plot(layer_weights[:,filter_num], color=np.random.rand(3,))
        plt.vlines(0,-0.5,0.5, color = "white")
        # set to true for first dense layer after concatenation
        if(False):
            plt.vlines([w_global_avg.shape[1], 
                        w_global_avg.shape[1] + w_dilated_global_avg.shape[1]],-0.5,0.5)
        plt.show()


def qq_plot(sample_1, sample_2, num_quantiles=100, axlabels=['sample 1', 'sample 2']):
    plt.scatter(np.quantile(sample_1, np.arange(0,1,1/num_quantiles)), 
           np.quantile(sample_2, np.arange(0,1,1/num_quantiles)))
    plt.axline((np.mean(sample_1), np.mean(sample_2)), slope = 1, color = "red")
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    plt.show()


def make_experiment_density_plots(ref_pred_ape, ref_phylo_ape, 
                         misspec_pred_ape,  misspec_phylo_ape, 
                         baseline_ape, 
                            xlabel = ["R0", "sample rate", "migration rate"],
                           plot_legend = ['random', 'CNN', 'CNN misspec', 'Phylo', 'Phylo misspec']):
    # ref for cnn and phylo, then misspecified for cnn and phylo
    
    colors = ['g', 'b', 'b', 'r', 'r']
    line_styles = [':','-','--', '-','--']
    
    # make density plots
    for i in range(0, ref_pred_ape.shape[1]):
        xlim_low = np.min(np.concatenate([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])]))
        xlim_high = np.max(np.concatenate([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])]))
        df = pd.DataFrame([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])])
        df.transpose().plot(kind = 'density',
                           style = line_styles,
                           color = colors,
                           xlim = [xlim_low-1, xlim_high+1])
        plt.xlabel(xlabel[i] + " log abs. % error ")
        plt.legend(plot_legend)
        plt.show()

        # make boxplots
        box = plt.boxplot([ref_pred_ape[:,i],
          misspec_pred_ape[:,i], 
          ref_phylo_ape[:,i],
          misspec_phylo_ape[:,i]],
                   labels = ['CNN true', 'CNN misspec', 
                            'phylo true', 'phylo misspec'], 
                          showfliers = False, widths = 0.9, patch_artist = True)
        plt.axline((0.5,0), slope = 0, color = "red")
        plt.ylabel('percent error (APE)')
        plt.title(xlabel[i])
        for box, color in zip(box['boxes'], colors[1:]):
            box.set_edgecolor(color)
            box.set_facecolor('w')
        plt.show()
        
        # make histograms        
        plt.hist((misspec_pred_ape[:,i]) - (misspec_phylo_ape[:,i]), bins = 20)
        plt.axline((0,0), slope = 1000000, color = "red")
        plt.xlabel('cnn APE - phylo APE')
        plt.title(xlabel[i])
        plt.show()

        # print summary stats

