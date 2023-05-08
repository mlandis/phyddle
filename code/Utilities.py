import argparse
import importlib

def load_config(config_fn, arg_overwrite=True):
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config')
    parser.add_argument('--cfg', dest='config_fn', type=str, help='Config file name')
    args = parser.parse_args()
    
    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn != None:
        config_fn = args.config_fn
    
    # config from filep
    m = importlib.import_module(config_fn)
    return m