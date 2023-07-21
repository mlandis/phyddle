#!/usr/bin/env python
"""
ModelLoader
===========
Defines a registry of recognized model types and variants. Also defines methods to quick-load
requested models as needed for a phyddle analysis.

Authors:   Michael Landis, Ammon Thompson
Copyright: (c) 2023, Michael Landis
License:   MIT
"""

# standard libraries
import importlib
import pandas as pd

# model string names and class names
model_registry = []
model_registry_names = ['model_name', 'class_name',    'description' ] 
model_registry.append( ['geosse',     'GeosseModel',   'Geographic State-dependent Speciation Extinction [GeoSSE]'] )
model_registry.append( ['sirm',       'SirmModel',     'Susceptible-Infected-Recovered-Migration [SIRM]'] )
model_registry = pd.DataFrame( model_registry, columns = model_registry_names)

# convert model_name into class_name through registry
def get_model_class(model_type):
    model_class_name = model_registry.class_name[ model_registry.model_name == model_type ].iat[0]
    MyModelModule = importlib.import_module('phyddle.Models.'+model_class_name)
    #cls = getattr(import_module('my_module'), 'my_class') 
    MyModelClass = getattr(MyModelModule, model_class_name)
    return MyModelClass

# print
def make_model_registry_str():
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
        MyModelModule = importlib.import_module('phyddle.Models.'+model_class)
        #MyModelClass = getattr(MyModelModule, model_class)
        variant_registry = MyModelModule.variant_registry
        for j in range(len(variant_registry)):
            variant_j = variant_registry.loc[j]
            variant_name = variant_j.variant_name
            variant_desc = variant_j.description
            s += ''.ljust(20, ' ') + variant_name.ljust(20, ' ') + variant_desc.ljust(40, ' ') + '\n'
        s += '\n'

    return s

def load(args):
    model_type = args['model_type']
    MyModelClass = get_model_class(model_type)
    return MyModelClass(args)

