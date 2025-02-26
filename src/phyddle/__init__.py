#!/usr/bin/env python
"""
__init__
===========
Initializes phyddle package and discovers all modules.

Authors:   Michael Landis, Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""

#import os
#import sys
from multiprocessing import set_start_method

__project__ = 'phyddle'
__version__ = '0.3.0'
__author__ = 'Michael Landis and Ammon Thompson'
__copyright__ = '(c) 2022-2025, Michael Landis and Ammon Thompson'
__citation__ = 'MJ Landis, A Thompson. 2024. phyddle: software for phylogenetic model exploration with deep learning. bioRxiv 2024.08.06.606717.'

# DEFAULT
CONFIG_DEFAULT_FN = 'config_default.py'
PHYDDLE_VERSION = __version__

# add local directory to system path
# allows import of local cfg
# ...but might switch over to simple tab-deilmited file
#sys.path.insert(0, os.getcwd())

if __name__ == '__main__':
    # set the start method
    set_start_method('fork')
