
import phyddle
import phyddle.utilities as util
import phyddle.simulate as sim
import phyddle.format as fmt
import phyddle.train as trn
import phyddle.estimate as est
import phyddle.plot as plt

# set random seed
import numpy as np
import os

#-----------------------------------------------------------------------------#

def inc(x):
    return x + 1

def test_inc():
    assert inc(3) == 4

#-----------------------------------------------------------------------------#

def test_onehot_encoding():
    s = util.convert_nexus_to_array('./data/sim.1.dat.nex', 'onehot', 333)
    print(s)
    assert s != ''

#-----------------------------------------------------------------------------#

#def test_script():
#    cmd_str = 'Rscript ./tests/sim_one.R ./tests/workspace/raw_data/test/sim.0'
#    cmd_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
#    assert 1 == 1

#-----------------------------------------------------------------------------#

def run_pipeline():

    my_args = util.load_config('config', arg_overwrite=True, args=[])
    
    proj = my_args['proj']
    sim_dir = my_args['sim_dir']
    fmt_dir = my_args['fmt_dir']

    ################
    # RUN PIPELINE #
    ################

    # Step 1: simulate training data
    my_sim = sim.load(my_args)
    my_sim.run()

    print(os.getcwd())
    print(os.listdir('./tests/workspace/simulate/test'))

    # Step 2: format training data into tensors
    my_fmt = fmt.load(my_args)
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

    phy_data_fn = f'{sim_dir}/{proj}/sim.0.phy_data.csv'
    phy_data = np.loadtxt(phy_data_fn, delimiter=',')
    phy_data_sum = round(np.sum(phy_data), ndigits=8)
    print(phy_data_sum)

    return phy_data_sum

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

