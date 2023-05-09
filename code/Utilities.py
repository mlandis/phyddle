# general libraries
import argparse
import importlib
import pandas as pd
from GeosseModel import *

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

#from Models.SirmModel import SirmModel
#from Models.GeosseModel import GeosseModel

# register models for retrieval
def make_model(mdl_args):    
    model_type = mdl_args['model_type']
    if model_type == 'geosse':      mdl = GeosseModel
    #elif model_type == 'sirm':      mdl = SirmModel
    #elif model_type == 'nichesse':    return NicheModel
    #elif model_type == 'birthdeath':  return BirthDeathModel
    #elif model_type == 'chromosse':   return ChromosseModel
    #elif model_type == 'classe':      return ClasseModel
    #elif model_type == 'musse':       return MusseModel
    else:                           return None
    return mdl(mdl_args)

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