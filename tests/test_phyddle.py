from phyddle import Utilities
from phyddle import ModelLoader
from phyddle import Simulating
from phyddle import Formatting
from phyddle import Learning
from phyddle import Predicting
from phyddle import Plotting

# set random seed
import numpy as np
import h5py
import os
import subprocess

#print(os.getcwd())
#print(os.listdir('.'))
#print(os.listdir('./workspace'))
#print(os.listdir('./workspace/raw_data/test'))

#-----------------------------------------------------------------------------#

def inc(x):
    return x + 1

def test_inc():
    assert inc(3) == 4

#-----------------------------------------------------------------------------#

def test_onehot_encoding():
    s = Utilities.convert_nexus_to_array('./data/sim.1.dat.nex', 'onehot', 333)
    print(s)
    assert s != ''

#-----------------------------------------------------------------------------#

#def test_script():
#    cmd_str = 'Rscript ./tests/sim_one.R ./tests/workspace/raw_data/test/sim.0'
#    cmd_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
#    assert 1 == 1

#-----------------------------------------------------------------------------#

def run_pipeline():

    my_args = Utilities.load_config('config', arg_overwrite=True, args=[])
    
    proj = my_args['proj']
    sim_dir = my_args['sim_dir']
    fmt_dir = my_args['fmt_dir']

    ################
    # RUN PIPELINE #
    ################

    # Step 1: simulate training data
    my_mdl = ModelLoader.load(my_args)
    my_sim = Simulating.load(my_args, my_mdl)
    my_sim.run()

    print('here i am')
    print(os.getcwd())
    print(os.listdir('./tests/workspace/raw_data/test'))
    #print(os.listdir('./tests/workspace/raw_data'))


    # Step 2: format training data into tensors
    my_fmt = Formatting.load(my_args)
    my_fmt.run()

    # Step 3: train network with training data
    #my_lrn = Learning.load(my_args)
    #my_lrn.run()

    # Step 4: predict for new dataset
    #my_prd = Predicting.load(my_args)
    #pred_prefix = f"{my_args['pred_dir']}/{my_args['proj']}/{my_args['pred_prefix']}"
    #my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyenc_csv=True)
    #my_prd.run()

    # Step 5: plot results
    #my_plt = Plotting.load(my_args)
    #my_plt.run()

    cdvs_fn = f'{sim_dir}/{proj}/sim.0.cdvs.csv'
    cdvs = np.loadtxt(cdvs_fn, delimiter=',')
    cdvs_sum = round(np.sum(cdvs), ndigits=8)
    print(cdvs_sum)

    return cdvs_sum

def test_run_pipeline():
    val = run_pipeline()
    assert val == 171.25643737


#test_run_pipeline()
#-----------------------------------------------------------------------------#

# tests needed
# - all types of phylostate tensor encoding
# - summ stat encoding
# - model loading
# - model setup, events, statespace, etc.

