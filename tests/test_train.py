"""
test_simulate
=============
Tests classes and methods for the Simulate step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.train as trn

#-----------------------------------------------------------------------------#

def load():
    my_args = util.load_config('config', arg_overwrite=True, args=[])
    my_trn = trn.load(my_args)
    #my_trn.run()
    return

def test_load():
	load()
	return

#-----------------------------------------------------------------------------#