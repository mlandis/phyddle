from phyddle import Utilities
from phyddle import ModelLoader
from phyddle import Simulating
from phyddle import Formatting
from phyddle import Learning
from phyddle import Predicting
from phyddle import Plotting

# set random seed
import numpy as np

#-----------------------------------------------------------------------------#

def inc(x):
    return x + 1

def test_inc():
    assert inc(3) == 4

#-----------------------------------------------------------------------------#

def test_onehot_encoding():
    s = Utilities.convert_nexus_to_array('data/sim.1.dat.nex', 'onehot', 333)
    print(s)
    assert s != ''

#-----------------------------------------------------------------------------#

def run_pipeline():
    return 1

def test_run_pipeline():
    assert run_pipeline() == 1

#-----------------------------------------------------------------------------#

# tests needed
# - all types of phylostate tensor encoding
# - summ stat encoding
# - model loading
# - model setup, events, statespace, etc.

