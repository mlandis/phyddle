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
