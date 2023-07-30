#!/usr/bin/env python
"""
__init__
===========
Initializes phyddle package and discovers all modules.

Authors:   Michael Landis, Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

from multiprocessing import set_start_method

__project__ = 'phyddle'
__version__ = '0.0.5'
__author__ = 'Michael Landis and Ammon Thompson'
__copyright__ = '(c) 2022-2023, Michael Landis and Ammon Thompson'
__citation__ = 'A Thompson, B Liebeskind, EJ Scully, MJ Landis. 2023. Deep learning approaches to viral phylogeography are fast and as robust as likelihood methods to model misspecification. bioRxiv 2023.02.08.527714.'


if __name__ == '__main__':
    # set the start method
    set_start_method('fork')
