#!/usr/bin/env python
"""
simulate
========
Defines classes and methods for the Simulate step, which generates large numbers
of simulated datasets (in parallel, if desired) that are later formatted and used
to train the neural network.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import gzip
import os
import re
import shutil
import subprocess
#import time

# external imports
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# phyddle imports
from phyddle import utilities

#------------------------------------------------------------------------------#

def load(args, mdl=None):
    """
    Load the appropriate simulator.

    This function returns an instance of a simulator class based on the
    `sim_method` key in the provided `args` dictionary. The supported simulators are
    `CommandSimulator` and `MasterSimulator`. If an unsupported value is provided,
    the function returns `None`.

    Args:
        args (dict): A dictionary containing configuration parameters for the
            simulators. Must include the key 'sim_method' which should have a value of
            either 'command' or 'master'.
        mdl (Model): The model instance that the simulator will operate on.

    Returns:
        Simulator: An instance of the `CommandSimulator` or `MasterSimulator` class,
            or `None` if an unsupported `sim_method` is provided.

    """

    # load object
    sim_method = args['sim_method']
    if sim_method == 'command':
        return CommandSimulator(args, mdl)
    elif sim_method == 'master':
        return MasterSimulator(args, mdl)
    else:
        return None

#------------------------------------------------------------------------------#

class Simulator:
    """
    A class representing a simulator.

    Args:
        args (dict): A dictionary containing the simulator arguments.
        mdl (str): The model for simulation.

    Attributes:
        args (dict): The simulator arguments.
        proj (str): The project name.
        verbose (bool): A flag indicating the verbosity level.
        sim_dir (str): The directory for simulation.
        start_idx (int): The start index.
        end_idx (int): The end index.
        stop_time (int): The stop time.
        num_proc (int): The number of processes.
        use_parallel (bool): A flag indicating whether to use parallel processing.
        sample_population (int): The sample population.
        min_num_taxa (int): The minimum number of taxa.
        max_num_taxa (int): The maximum number of taxa.
        sim_logging (bool): A flag indicating whether to enable simulation logging.
        rep_idx (list): A list of replicate indices.
        save_params (bool): A flag indicating whether to save parameters.
        model (str): The model for simulation.
        logger (utilities.Logger): An instance of the logger for logging.

    """
    def __init__(self, args, mdl):
        """
        Initializes a new instance of the Simulator class.

        Args:
            args (dict): A dictionary containing the simulator arguments.
            mdl (str): The model for simulation.

        """
        self.set_args(args)
        #self.sim_command  = 'echo \"phyddle.Simulator.sim_command undefined in derived class!\"' # do nothing
        self.model = mdl
        self.logger = utilities.Logger(args)
        
        return

    def set_args(self, args):
        """
        Sets the simulator arguments.

        Args:
            args (dict): A dictionary containing the simulator arguments.

        """
        # simulator arguments
        self.args              = args
        self.verbose           = args['verbose']
        self.sim_dir           = args['sim_dir']
        self.sim_proj          = args['sim_proj']
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.stop_time         = args['stop_time']
        self.num_proc          = args['num_proc']
        self.use_parallel      = args['use_parallel']
        self.sample_population = args['sample_population']
        self.min_num_taxa      = args['min_num_taxa']
        self.max_num_taxa      = args['max_num_taxa']
        self.sim_logging       = args['sim_logging']
        self.rep_idx           = list(range(self.start_idx, self.end_idx))
        self.save_params       = False

        self.sim_proj_dir      = f'{self.sim_dir}/{self.sim_proj}'
        return

    def make_settings_str(self, idx, mtx_size):
        """
        Generates the settings string.

        Args:
            idx (int): The replicate index.
            mtx_size (str): The taxon category.

        Returns:
            str: The generated settings string.

        """
        s =  'setting,value\n'
        s += f'sim_proj,{self.sim_proj}\n'
        s += f'model_name,{self.model_type}\n'
        s += f'model_variant,{self.model_variant}\n'
        s += f'replicate_index,{idx}\n'
        s += f'taxon_category,{mtx_size}\n'
        s += f'sim_method,{self.sim_method}\n'
        return s

    def run(self):
        """
        This method runs the simulation process.

        Returns:
            list: The result of the simulation process.
        """
        #if self.verbose:
        #    print( utilities.phyddle_info('sim', self.sim_proj, None, self.sim_dir) )
        utilities.print_step_header('sim', None, self.sim_proj_dir, verbose=self.verbose)

        # prepare workspace
        os.makedirs( self.sim_dir + '/' + self.sim_proj, exist_ok=True )
        # dispatch jobs

        #if self.verbose: print(utilities.phyddle_str('▪ simulating raw data ...'))
        utilities.print_str('▪ simulating raw data ...', verbose=self.verbose)
        
        if self.use_parallel:
            res = Parallel(n_jobs=self.num_proc)(delayed(self.sim_one)(idx) for idx in tqdm(self.rep_idx))
        else:
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx) ]

        utilities.print_str('... done!', verbose=self.verbose)
        return res

    # main simulation function (looped)
    def sim_one(self, idx):
        """
        This method runs a single simulation iteration.

        Args:
            idx (int): The index of the simulation iteration.

        Returns:
            None
        """
        # improve numpy output style (move to utilities??)
        NUM_DIGITS = 10
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')}, precision=NUM_DIGITS)
        
        # make filenames
        out_path  = self.sim_dir + '/' + self.sim_proj + '/sim'
        tmp_fn    = out_path + '.' + str(idx)
        param_mtx_fn = tmp_fn + '.param_col.csv'
        param_vec_fn = tmp_fn + '.param_row.csv'
        
        # refresh model and update XML string (redraw parameters, etc.)
        self.refresh_model(idx)
        
        # record labels (simulating parameters)
        if self.save_params:
            param_mtx_str,param_vec_str = utilities.param_dict_to_str(self.model.params)
            utilities.write_to_file(param_mtx_str, param_mtx_fn)
            utilities.write_to_file(param_vec_str, param_vec_fn)
            
        # delegate simulation to derived Simulator
        self.sim_one_custom(idx)

        # done!
        return

    def refresh_model(self, idx):
        """
        This method refreshes the simulation model for a given iteration.

        Args:
            idx (int): The index of the simulation iteration.

        Returns:
            None
        """
        self.model.set_model(idx)
        self.start_state   = self.model.start_state
        self.start_sizes   = self.model.start_sizes
        self.df_events     = self.model.df_events
        self.refresh_model_custom(idx)
        return

    def sim_one_custom(self, idx):
        """
        Simulates a custom operation for a specific index.

        Parameters:
        - idx: The index representing the specific operation to simulate.

        Returns:
        - The result of simulating the custom operation.

        Raises:
        - NotImplementedError: If the method is not implemented.

        """
        return NotImplementedError
    
    def refresh_model_custom(self, idx):
        """
        Refreshes a custom model for a specific index.

        Parameters:
        - idx: The index representing the specific model to refresh.

        Returns:
        - The result of refreshing the custom model.

        Raises:
        - NotImplementedError: If the method is not implemented.

        """
        return NotImplementedError    

#------------------------------------------------------------------------------#

###################################
# Generic CLI simulator interface #
###################################

class CommandSimulator(Simulator):
    """
    A class that represents a command simulator.

    Args:
        args (dict): A dictionary containing arguments.
        mdl (Model): An instance of the Model class.

    Attributes:
        sim_command (str): The command to run for simulation.

    """
    def __init__(self, args, mdl):
        super().__init__(args, mdl)
        return
    
    def set_args(self, args):
        """
        Sets the arguments for the simulator.

        Args:
            args (dict): A dictionary containing arguments.

        """
        super().set_args(args)
        self.sim_command = args['sim_command']
        return

    def sim_one_custom(self, idx):
        """
        Runs a generic simulation job.

        Args:
            idx (int): The index of the simulation job.

        """

        # get filesystem info for generic job
        out_path   = f'{self.sim_dir}/{self.sim_proj}/sim'
        tmp_fn     = f'{out_path}.{idx}'
        dat_fn     = tmp_fn + '.dat.nex'
        cmd_log_fn = tmp_fn + '.sim_command.log'

        # run generic job
        # example:
        #   self.sim_command == 'rb sim_one.Rev --args'
        #   tmp_fn == '../workspace/raw_data/Rev_example/sim.0'
        cmd_str = f'{self.sim_command} {tmp_fn}'

        num_attempt = 10
        valid = False
        while not valid and num_attempt > 0:
            try:
                cmd_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
                utilities.write_to_file(cmd_out, cmd_log_fn)
                valid = True
            except subprocess.CalledProcessError:
                self.logger.write_log('sim', f'simulation {idx} failed to generate a valid dataset')
                num_attempt -= 1
                valid = False
                #print(f'error for rep_idx={idx}')

        return
    
    def refresh_model_custom(self, idx):
        """
        Refreshes the custom model with a specific index.

        Args:
            idx (int): The index of the custom model.

        """
        return


#------------------------------------------------------------------------------#

##############################
# MASTER simulator interface #
##############################

class MasterSimulator(Simulator):
    def __init__(self, args, mdl):
        """
        Initializes a new instance of the MasterSimulator class.

        Args:
            args (list): A list of arguments for the simulator.
            mdl: The model to be used in the simulation.

        Returns:
            None.
        """
        # call base constructor
        super().__init__(args, mdl)
        self.save_params = True
        return

    def refresh_model_custom(self, idx):
        """
        Refreshes the custom model for the simulation.

        Args:
            idx (int): The index of the model to be refreshed.

        Returns:
            None.
        """
        self.reaction_vars = self.make_reaction_vars()
        self.xml_str       = self.make_xml(idx)
        self.sim_command   = 'beast {sim_dir}/{proj}/sim.{idx}.xml'.format(sim_dir=self.sim_dir, proj=self.sim_proj, idx=idx)


    def sim_one_custom(self, idx):
        """
        Simulates a custom scenario.
        
        Parameters:
            self (object): The calling object.
            idx (int): The index of the scenario.
            
        Returns:
            None
        """
        out_path   = self.sim_dir + '/' + self.sim_proj + '/sim'
        tmp_fn     = out_path + '.' + str(idx)

        beast_fn   = tmp_fn + '.beast.log'
        xml_fn     = tmp_fn + '.xml'
        json_fn    = tmp_fn + '.json'
        phy_nex_fn = tmp_fn + '.phy.nex' # annotated tree, no mtx
        dat_nex_fn = tmp_fn + '.dat.nex' # no tree, character mtx
        tre_fn     = tmp_fn + '.tre'

        # make XML file
        xml_str = self.xml_str
        utilities.write_to_file(xml_str, xml_fn)

        # run BEAST job
        cmd_str = self.sim_command
        beast_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
        utilities.write_to_file(beast_out, beast_fn)

        # this code should convert to standard nexus
        # convert MASTER tree to standard nexus
        #taxon_states,
        # state space
        int2vec = self.model.states.int2vec
        nexus_str = utilities.convert_phy2dat_nex(phy_nex_fn, int2vec)
        utilities.write_to_file(nexus_str, dat_nex_fn)


        # logging clean-up
        if self.sim_logging == 'clean':
            for x in [ xml_fn, beast_fn, json_fn ]:
                if os.path.exists(x):
                    os.remove(x)
        elif self.sim_logging == 'compress':
            for x in [ xml_fn, beast_fn, json_fn ]:
                if os.path.exists(x):
                    with open(x, 'rb') as f_in:
                        with gzip.open(x+'.gz', 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)        
                    os.remove(x)
        elif self.sim_logging == 'verbose':
            pass
            # do nothing
        return
    
    def make_reaction_vars(self):
        """
        Generate a dictionary of reaction variables and their corresponding counts.

        Returns:
            dict: A dictionary containing reaction variable names as keys and the number
            of locations/substates associated with each variable as values.
        """
        qty = {}
        # get list of all reaction variables
        for s in self.df_events.reaction:
            # regular expression extracts all Variable and Element records
            # encoded in the LHS (.ix) and RHS (.ix) of df_events
            toks = re.findall( r'([0-9]*([A-Za-z])(\[[0-9]+\])?)', s)
            #toks = re.findall( r'([0-9]*([A-Za-z])(\[[0-9]+\])?)(:1)?', s)
            for v in toks:
                var_name = v[1]
                var_idx = v[2]
                if var_idx != '':
                    if var_name not in qty:
                        qty[var_name] = set([var_idx])
                    else:
                        qty[var_name].add(var_idx)
                else:
                    qty[var_name] = set()

        # determine how many locations/substates per reaction label
        reaction_vars = {}
        for k,v in qty.items():
            reaction_vars[k] = len(v)
            
        return reaction_vars


    def make_xml(self, idx):
        """
        Creates an XML specification string for a simulation.

        Parameters:
        - idx (int): The index of the simulation.

        Returns:
        - xml_spec_str (str): The XML specification string for the simulation.
        """
        # file names
        newick_fn = '{sim_dir}/{proj}/sim.{idx}.tre'.format(sim_dir=self.sim_dir, proj=self.sim_proj, idx=idx)
        nexus_fn  = '{sim_dir}/{proj}/sim.{idx}.phy.nex'.format(sim_dir=self.sim_dir, proj=self.sim_proj, idx=idx)
        json_fn   = '{sim_dir}/{proj}/sim.{idx}.json'.format(sim_dir=self.sim_dir, proj=self.sim_proj, idx=idx)

        # state space
        xml_statespace = ''
        for k,v in self.reaction_vars.items():
            if v == 0:
                xml_statespace += "<populationType spec='PopulationType' typeName='{k}' id='{k}'/>\n".format(k=k)
            elif v > 0:
                xml_statespace += "<populationType spec='PopulationType' typeName='{k}' id='{k}' dim='{v}'/>\n".format(k=k,v=v)
        
        # reaction groups
        xml_events = ''
        groups = set(self.df_events.group)
        for g in groups:
            xml_events += "<reactionGroup spec='ReactionGroup' reactionGroupName='{g}'>\n".format(g=g)
            #for row in self.events[ self.events.group == g ]:
            for i in range(0, len(self.df_events[ self.df_events.group == g ])):
                row = self.df_events[ self.df_events.group == g ].iloc[i]
                rate     = row['rate']
                name     = row['name']
                reaction = row['reaction']
                xml_events += "\t<reaction spec='Reaction' reactionName='{n}' rate='{r}'>\n\t\t{x}\n\t</reaction>\n".format(n=name, r=rate, x=reaction)
            xml_events += "</reactionGroup>\n"
            xml_events += '\n'

        # starting state of simulation
        #start_state = 'S[{i}]'.format(i=start_index)
        xml_init_state = "<initialState spec='InitState'>\n"
        for k,v in self.start_sizes.items():
            for i,y in enumerate(v):
                if y > 0:
                    xml_init_state += "\t<populationSize spec='PopulationSize' size='{y}'>\n".format(y=y)
                    xml_init_state += "\t\t<population spec='Population' type='@{k}' location='{i}'/>\n".format(k=k,i=i)
                    xml_init_state += "\t</populationSize>\n"
        for k,v in self.start_state.items():
            xml_init_state += "\t<lineageSeedMultiple spec='MultipleIndividuals' copies='1'>\n"
            xml_init_state += "\t\t<population spec ='Population' type='@{k}' location='{v}'/>\n".format(k=k, v=v)
            xml_init_state += "\t</lineageSeedMultiple>\n"

        xml_init_state += "</initialState>\n"

        # sim conditions
        xml_sim_conditions = ""
        xml_sim_conditions += f"<lineageEndCondition spec='LineageEndCondition' nLineages='{self.max_num_taxa}' alsoGreaterThan='true' isRejection='false'/>\n" #.format(stop_ceil_sizes=self.stop_ceil_sizes)
        # xml_sim_conditions += f"<lineageEndCondition spec='LineageEndCondition' nLineages='{self.min_num_taxa}' alsoGreaterThan='false' isRejection='false'/>\n" #.format(stop_floor_sizes=self.stop_floor_sizes)
        # xml_sim_conditions += "<postSimCondition spec='LeafCountPostSimCondition' nLeaves='10' exact='false' exceedCondition='true'/>\n"

        # post-processing filter
        sample_population = self.sample_population #settings['sample_population']
        xml_filter = ''
        for k in sample_population:
            xml_filter += "<inheritancePostProcessor spec='LineageFilter' populationName='{k}' reverseTime='false' discard='false' leavesOnly='true' noClean='true'/>\n".format(k=k)

        # stop time
        stop_time = self.stop_time

        # generate entire XML specification
        xml_spec_str = '''\
<beast version='2.0' namespace='master:master.model:master.steppers:master.conditions:master.postprocessors:master.outputs'>

<run spec='InheritanceEnsemble'
    verbosity='3'
    nTraj='1'
    nSamples='{num_samples}'
    samplePopulationSizes='{sample_pop}'
    simulationTime='{stop_time}'
    maxConditionRejects='1'>

<model spec='Model'>

{xml_statespace}

{xml_events}

</model>

{xml_init_state}

{xml_sim_conditions}

{xml_filter}

<output spec='NewickOutput' collapseSingleChildNodes='true' fileName='{newick_fn}'/>
<output spec='NexusOutput' fileName='{nexus_fn}'/>
<output spec='JsonOutput' fileName='{json_fn}' />

</run>
</beast>
'''.format(xml_statespace=xml_statespace,
           xml_events=xml_events,
           xml_init_state=xml_init_state,
           xml_sim_conditions=xml_sim_conditions,
           xml_filter=xml_filter,
           newick_fn=newick_fn, nexus_fn=nexus_fn, json_fn=json_fn,
           num_samples=1000,
           sample_pop='true',
           stop_time=stop_time)
        
        return xml_spec_str
        #return xml_spec_str
