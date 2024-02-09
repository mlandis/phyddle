#!/usr/bin/env python
"""
estimate
========
Defines classes and methods for the Estimate step, which loads a pre-trained
network and uses it to generate new estimates, e.g. estimate model parmaeters
for a new empirical dataset.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
#import os

# phyddle imports
#from phyddle import utilities as util

class PipelineStep:
    """
    Class for phyddle steps.
    """

    def __init__(self, args):
        return

    def set_args(self, args):
        return
    
    def run(self):
        self.prepare_filesystem()
        self.load_input()
        self.run()
        self.process_results()
        self.save_results()
        return

    def prepare_variables(self):
        return

    def prepare_filesystem(self):
        return

    def load_input(self):
        return
    
    def process_output(self):
        return
    
    def save_output(self):
        return