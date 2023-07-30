#!/usr/bin/env python
"""
model_loader
============
Defines a registry of recognized model types and variants. Also defines methods
to quick-load requested models as needed for a phyddle analysis.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard libraries
import importlib
import pandas as pd
from itertools import combinations,chain

# model string names and class names
model_registry = []
model_registry_names = ['model_name', 'class_name',    'description' ] 
model_registry.append( ['geosse',     'GeosseModel',   'Geographic State-dependent Speciation Extinction [GeoSSE]'] )
model_registry.append( ['sirm',       'SirmModel',     'Susceptible-Infected-Recovered-Migration [SIRM]'] )
model_registry = pd.DataFrame( model_registry, columns = model_registry_names)

def load(args):
    """
    Generates a Google-style docstring for the load function.
    
    Parameters:
        args (dict): A dictionary containing the arguments required for model loading.
    
    Returns:
        obj: An instance of the loaded model.
    """
    model_type = args['model_type']
    MyModelClass = get_model_class(model_type)
    return MyModelClass(args)

# convert model_name into class_name through registry
def get_model_class(model_type):
    """
    Returns the corresponding model class based on the given model_type.

    Parameters:
    - model_type (str): The type of the model to be retrieved.

    Returns:
    - MyModelClass: The class object of the corresponding model.
    """
    model_class_name = model_registry.class_name[ model_registry.model_name == model_type ].iat[0]
    MyModelModule = importlib.import_module('models.'+model_class_name)
    #cls = getattr(import_module('my_module'), 'my_class') 
    MyModelClass = getattr(MyModelModule, model_class_name)
    return MyModelClass

# print
def make_model_registry_str():
    """
    Generates a Google-style docstring for the make_model_registry_str function.
    
    Returns:
        str: The formatted model registry string.
    """
    s = ''
    # header
    s += 'Type'.ljust(20, ' ') + 'Variant'.ljust(20, ' ') + 'Description'.ljust(40, ' ') + '\n'
    s += ''.ljust(60, '=') + '\n'
    # types
    for i in range(len(model_registry)): 
        model_i = model_registry.loc[i]
        model_name = model_i.model_name
        model_desc = model_i.description
        s += model_name.ljust(20, ' ') + '--'.ljust(20, ' ') + model_desc.ljust(40, ' ') + '\n'
        # variants per type
        model_class = model_i.class_name
        MyModelModule = importlib.import_module('models.'+model_class)
        #MyModelClass = getattr(MyModelModule, model_class)
        variant_registry = MyModelModule.variant_registry
        for j in range(len(variant_registry)):
            variant_j = variant_registry.loc[j]
            variant_name = variant_j.variant_name
            variant_desc = variant_j.description
            s += ''.ljust(20, ' ') + variant_name.ljust(20, ' ') + variant_desc.ljust(40, ' ') + '\n'
        s += '\n'

    return s


# model events
class Event:
    """
    Event objects define an event for a Poisson process with discrete-valued
    states, such as continuous-time Markov processes. Note, that
    phylogenetic birth-death models and SIR models fall into this class.
    The Event class was originally designed for use with chemical
    reaction simulations using the MASTER plugin in BEAST.
    """
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
        """
        Create an Event object.

        Args:
            idx (dict): A dictionary containing the indices of the event.
            r (float): The rate of the event.
            n (str): The name of the event.
            g (str): The reaction group of the event.
            ix (list): The reaction quantities (reactants) before the event.
            jx (list): The reaction quantities (products) after the event.
        """
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
        return
        
    # make print string
    def make_str(self):
        """
        Creates a string representation of the event.

        Returns:
            str: The string representation of the event.
        """
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    
    # representation string
    def __repr__(self):
        """
        Returns the representation of the event.

        Returns:
            str: The representation of the event.
        """
        return self.make_str()
    
    # print string
    def __str__(self):
        """
        Returns the string representation of the event.

        Returns:
            str: The string representation of the event.
        """
        return self.make_str()


# state space
class States:
    """
    States objects define the state space that a model operates upon. Event
    objects define transition rates and patterns with respect to States. The
    central purpose of States is to manage different representations of
    individual states in the state space, e.g. as integers, strings, vectors.
    """
    def __init__(self, lbl2vec):
        """
        Create a States object.

        Args:
            lbl2vec (dict): A dictionary with labels (str) as keys and vectors
                            of states (int[]) as values.
        """
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
       
        # done
        return

    def make_str(self):
        """
        Creates a string representation of the state space.

        Returns:
            str: The string representation of the state space.
        """
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
        """
        Returns the representation of the state space.

        Returns:
            str: The representation of the state space.
        """
        return self.make_str()

    # print string
    def __str__(self):
        """
        Returns the string representation of the state space.

        Returns:
            str: The string representation of the state space.
        """
        return self.make_str()
    

def events2df(events):
    """
    Convert a list of Event objects to a pandas DataFrame.

    This function takes a list of Event objects and converts it into a pandas DataFrame. Each Event object represents a row in the resulting DataFrame, with the Event attributes mapped to columns.

    Args:
        events (list): A list of Event objects.

    Returns:
        pandas.DataFrame: The resulting DataFrame with columns 'name', 'group', 'i', 'j', 'k', 'reaction', and 'rate'.
    """
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
    """
    Convert a States object to a pandas DataFrame.

    This function takes a States object and converts it into a pandas DataFrame. The States object contains information about the state space, and the resulting DataFrame has columns 'lbl', 'int', 'set', and 'vec', representing the labels, integer representations, set representations, and vector representations of the states, respectively.

    Args:
        states (States): The States object to convert to a DataFrame.

    Returns:
        pandas.DataFrame: The resulting DataFrame with columns 'lbl', 'int', 'set', and 'vec'.
    """
    df = pd.DataFrame({
        'lbl' : states.int2lbl,
        'int' : states.int2int,
        'set' : states.int2set,
        'vec' : states.int2vec
    })
    return df

def sort_binary_vectors(binary_vectors):
    """
    Sorts a list of binary vectors.

    The binary vectors are sorted first based on the number of "on" bits, and then from left to right in terms of which bits are "on".

    Args:
        binary_vectors (List[List[int]]): The list of binary vectors to be sorted.

    Returns:
        List[List[int]]: The sorted list of binary vectors.
    """
    def count_ones(binary_vector):
        """
        Counts the number of "on" bits in a binary vector.

        Args:
            binary_vector (List[int]): The binary vector.

        Returns:
            int: The count of "on" bits.
        """
        return sum(binary_vector)

    sorted_vectors = sorted(binary_vectors, key=count_ones)

    for i in range(len(sorted_vectors)):
        for j in range(i+1, len(sorted_vectors)):
            if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
                for k in range(len(sorted_vectors[i])):
                    if sorted_vectors[i][k] != sorted_vectors[j][k]:
                        if sorted_vectors[j][k] > sorted_vectors[i][k]:
                            sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
                        break

    return sorted_vectors

def powerset(iterable):
    """
    Generates all possible subsets (powerset) of the given iterable.

    Args:
        iterable: An iterable object.

    Returns:
        generator: A generator that yields each subset.
    """
    s = list(iterable)  # Convert the iterable to a list
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))