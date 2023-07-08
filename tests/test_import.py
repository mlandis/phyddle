#import ../code/ModelLoader
#import Simulating
#import Formatting
#import Learning
#import Predicting
#import ../code/Plotting
from phyddle import Utilities

def inc(x):
    return x + 1

def test_inc():
    assert inc(3) == 4

def one_hot_encoding(dat_fn, num_states):
    s = Utilities.convert_nexus_to_one_hot(dat_fn, num_states)
    return s

s = one_hot_encoding('data/sim.1.dat.nex', 3)
print(s)


