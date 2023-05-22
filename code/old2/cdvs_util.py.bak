#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import argparse
import random

from ete3 import Tree

#max_len = 501

TURN_ONE = 'turn_one'

# the information on state is saved as 't_s' in the newick tree
T_S = 't_s'
STATE = 'state'
DIVERSIFICATION_SCORE = 'diversification_score'

sys.setrecursionlimit(100000)


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


def name_tree(tr):
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


def add_dist_to_root(tr):
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

# #nexus
#Begin trees;
#tree TREE_0 = ((1[&type="B",reaction="Extinction",time=3.6232301233156976]:1.2142655324328055,(2[&type="B",time=10.0]:3.459208574144607,(3[&type="B",reaction="Extinction",time=6.868495320534605]:0.04404772822342995,((4[&type="B",time=10.0]:1.591682207985743,5[&type="B",reaction="Extinction",time=9.718023692584314]:1.3097059005700569)[&type="B",reaction="WithinRegionSpeciation",time=8.408317792014257]:1.395484627131916,6[&type="B",reaction="Extinction",time=7.337444843536313]:0.3246116786539721)[&type="B",reaction="WithinRegionSpeciation",time=7.012833164882341]:0.18838557257116584)[&type="B",reaction="WithinRegionSpeciation",time=6.824447592311175]:0.2836561664557822)[&type="B",reaction="WithinRegionSpeciation",time=6.540791425855393]:4.131826834972501)[&type="B",reaction="WithinRegionSpeciation",time=2.408964590882892]:1.5267870397153571,(7[&type="B",reaction="Extinction",time=2.98118850738119]:0.8725488577697549,(((8[&type="B",reaction="Extinction",time=9.850924475820728]:2.4517510454113083,9[&type="B",time=10.0]:2.6008265695905806)[&type="B",reaction="WithinRegionSpeciation",time=7.399173430409419]:4.033746617591738,10[&type="B",reaction="Extinction",time=4.670202523679139]:1.304775710861457)[&type="B",reaction="WithinRegionSpeciation",time=3.365426812817682]:0.3081968127823407,11[&type="B",reaction="Extinction",time=3.1071117381661546]:0.049881738130813424)[&type="B",reaction="WithinRegionSpeciation",time=3.057230000035341]:0.9485903504239062)[&type="B",reaction="WithinRegionSpeciation",time=2.108639649611435]:1.2264620984439)[&type="B",reaction="WithinRegionSpeciation",time=0.8821775511675349]:0.8821775511675349;
#End;

# [&&NHX:conf=0.01:name=INTERNAL]

# this newick string from this regex does not work, despite passing visual check
# x1=re.sub(r'&', r'&&NHX:', x); x2=re.sub(r'\"', '', x1);x3=re.sub(r',([a-z])', r':\1', x2);x3

def make_cdvs(tree_fn, max_len, states, state_labels):

    file = open(tree_fn, mode="r")
    
    tree_str = file.read()
    
    tree = Tree(tree_str, format=1)

    attach_tip_states(tree, states)
    #set_attribs(tree)
    name_tree(tree)

    #print('max_len ==>', max_len)
    #print('states ==>', states)

    # rescale tree to average branch length of 1
    # measure average branch length
    rescale_factor = get_average_branch_length(tree)

    # rescale tree
    rescale_tree(tree, rescale_factor)

    # add dist to root attribute
    tree, tr_height = add_dist_to_root(tree)

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
    complete_info = np.vstack( [tips_info, node_info] )
    
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
