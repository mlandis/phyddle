import numpy as np
import pandas as pd
import scipy as sp
import random
import string
import itertools
import re


def make_symm(m):
    d = np.diag(m)
    m = np.triu(m)
    m = m + m.T
    np.fill_diagonal(m, d)
    return m

# convert
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

# model events
class Event:
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, x=None, ix=None, jx=None):
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
        self.reaction = x
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


# model itself
class MasterXmlGenerator:
    # initialization
    def __init__(self, events, states, settings=None):
        self.states = states
        self.events = events
        self.settings = settings
        self.reaction_vars = self.make_reaction_vars()
    
    def make_reaction_vars(self):
        qty = {}
        for s in self.events.reaction:
            toks = re.findall( r'([0-9]*([A-Za-z])(\[[0-9]+\])?)', s)
            for v in toks:
                var_name = v[1]
                var_idx = v[2]
                if var_idx != '':
                    if var_name not in qty:
                        qty[var_name] = set([var_idx])
                    else:
                        qty[var_name].add(var_idx)
                else:
                    qty[var_name] = set()

        reaction_vars = {}
        for k,v in qty.items():
            reaction_vars[k] = len(v)
            
        return reaction_vars


    def make_xml(self, max_taxa, newick_fn, nexus_fn, json_fn):
    
        # states
        xml_statespace = ''
        for k,v in self.reaction_vars.items():
            if v == 0:
                xml_statespace += "<populationType spec='PopulationType' typeName='{k}' id='{k}'/>\n".format(k=k)
            elif v > 0:
                xml_statespace += "<populationType spec='PopulationType' typeName='{k}' id='{k}' dim='{v}'/>\n".format(k=k,v=v)
        #states = self.statespace.int2int
        #for st in self.states.int:
        #    xml_events += "<populationType spec='PopulationType' typeName='S[{st}]' id='{st}'/>\n".format(st=st)
        #xml_statespace += "<populationType spec='PopulationType' typeName='S' id='S' dim='{ns}'/>".format(ns=len(self.states.int))

        # get groups
        #groups = set([ e.group for e in events ])
        xml_events = ''
        groups = set(self.events.group)
        for g in groups:
            xml_events += "<reactionGroup spec='ReactionGroup' reactionGroupName='{g}'>\n".format(g=g)
            #for row in self.events[ self.events.group == g ]:
            for i in range(0, len(self.events[ self.events.group == g ])):
                row = self.events[ self.events.group == g ].iloc[i]
                rate     = row['rate']
                name     = row['name']
                reaction = row['reaction']
                xml_events += "\t<reaction spec='Reaction' reactionName='{n}' rate='{r}'>\n\t\t{x}\n\t</reaction>\n".format(n=name, r=rate, x=reaction)
            xml_events += "</reactionGroup>\n"
            xml_events += '\n'

        # init state
        # NOTE: uniform root state sampling is not ideal
        start_states = [0] * len(self.states.int)
        start_index = random.sample( list(self.states.int), 1)[0]
        start_states[start_index] = 1
        
        #start_state = 'S[{i}]'.format(i=start_index)
        xml_init_state = "<initialState spec='InitState'>\n"
        for k,v in self.reaction_vars.items():
            if v == 0:
                xml_init_state += "\t<lineageSeedMultiple spec='MultipleIndividuals' copies='{k}'>\n".format(k=start_states[i])
                xml_init_state += "\t\t<population spec ='Population' type='@{k}'/>\n".format(k=k)
                xml_init_state += "\t</lineageSeedMultiple>\n"
            else:
                for i in range(v):
                    xml_init_state += "\t<lineageSeedMultiple spec='MultipleIndividuals' copies='{k}'>\n".format(k=start_states[i])
                    xml_init_state += "\t\t<population spec ='Population' type='@{k}' location='{i}'/>\n".format(k=k,i=i)
                    xml_init_state += "\t</lineageSeedMultiple>\n"
        xml_init_state += "</initialState>\n"
        
        # sim conditions
        xml_sim_conditions = ""
        xml_sim_conditions += "<lineageEndCondition spec='LineageEndCondition' nLineages='{max_taxa}' alsoGreaterThan='true' isRejection='false'/>\n".format(max_taxa=max_taxa)
        xml_sim_conditions += "<lineageEndCondition spec='LineageEndCondition' nLineages='0' alsoGreaterThan='false' isRejection='false'>\n"
        for k,v in self.reaction_vars.items():
            if v == 0:
                xml_sim_conditions += "\t\t<population spec ='Population' type='@{k}'/>\n".format(k=k)
            else:
                for i in range(v):
                    xml_sim_conditions += "\t\t<population spec ='Population' type='@{k}' location='{i}'/>\n".format(k=k, i=i)
        # for i in self.states.int:
        #     xml_sim_conditions += "\t\t<population spec ='Population' type='@S' location='{i}'/>\n".format(i=i)
        xml_sim_conditions += "</lineageEndCondition>\n"
        xml_sim_conditions += "<postSimCondition spec='LeafCountPostSimCondition' nLeaves='10' exact='false' exceedCondition='true'/>\n"

        # generate entire XML specification
        xml_spec_str = '''\
<beast version='2.0' namespace='master:master.model:master.steppers:master.conditions:master.postprocessors:master.outputs'>

<run spec='InheritanceEnsemble'
    verbosity='3'
    nTraj='1'
    nSamples='{num_samples}'
    samplePopulationSizes='{sample_pop}'
    simulationTime='10'
    maxConditionRejects='1'>

<model spec='Model'>

{xml_statespace}

{xml_events}

</model>

{xml_init_state}

{xml_sim_conditions}

<output spec='NewickOutput' collapseSingleChildNodes='true' fileName='{newick_fn}'/>
<output spec='NexusOutput' fileName='{nexus_fn}'/>
<output spec='JsonOutput' fileName='{json_fn}' />

</run>
</beast>
'''.format(xml_statespace=xml_statespace, xml_events=xml_events, xml_init_state=xml_init_state, xml_sim_conditions=xml_sim_conditions, newick_fn=newick_fn, nexus_fn=nexus_fn, json_fn=json_fn, max_taxa=max_taxa, num_samples=1, sample_pop='false')
        self.xml_spec_str = xml_spec_str
        return
        #return xml_spec_str