"""
Logging
==========
Defines classes and methods to manage Logging of pipeline steps.

Authors:   Michael Landis, Ammon Thompson
Copyright: (c) 2023, Michael Landis
License:   MIT
"""

# - every time phyddle pipeline step run against workspace step/proj
# - create file if does not exist
# - Log level 0
# 	- log directory, timestamp, and command
# 	- step settings and errors
# - Log level 1: 
# 	- log project settings after config and CLI settings
# 	- log python version and runtime packages
	
import os
import sys
from datetime import datetime
import pkg_resources

class Logger:

    def __init__(self, args):
        
        # collect info from args        
        self.args        = args
        self.arg_str     = self.make_arg_str()
        self.job_id      = self.args['job_id']
        self.log_dir     = self.args['log_dir']
        self.proj        = self.args['proj']

        # collect other info and set constants
        self.pkg_name    = 'phyddle'
        self.version     = pkg_resources.get_distribution(self.pkg_name).version
        self.command     = ' '.join(sys.argv)
        self.date_obj    = datetime.now()
        self.date_str    = self.date_obj.strftime("%y%m%d_%H%M%S")
        self.max_lines   = 1e5

        # filesystem
        self.base_fn     = f'{self.pkg_name}_{self.version}_{self.date_str}'
        self.base_dir    = f'{self.log_dir}/{self.proj}'
        self.base_fp     = f'{self.base_dir}/{self.base_fn}' 
        self.fn_dict    = {
            'run' : f'{self.base_fp}.run.log',
            'sim' : f'{self.base_fp}.sim.log',
            'fmt' : f'{self.base_fp}.fmt.log',
            'lrn' : f'{self.base_fp}.lrn.log',
            'prd' : f'{self.base_fp}.prd.log',
            'plt' : f'{self.base_fp}.plt.log'
        }

        self.save_run_log()
        
        return

    def make_arg_str(self):
        s = ''
        for k,v in self.args.items():
            s += f'{k} = {v}\n'
        return s

    def save_log(self, step):

        if step == 'run':
            self.save_run_log()
        return

    def save_run_log(self):

        os.makedirs(self.base_dir, exist_ok=True)
        s = self.make_run_log()
        fn = self.fn_dict['run']
        f = open(fn, 'w')
        f.write(s)
        f.close()

        return
    
    def make_run_log(self):
        s = ''
        s += f'version = {self.version}\n'
        s += f'date = {self.date_str}\n'
        s += f'command = {self.command}\n'
        s += self.make_arg_str()
        return s