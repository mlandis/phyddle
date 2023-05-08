import argparse
import importlib

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
