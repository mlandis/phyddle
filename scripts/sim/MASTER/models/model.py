#!/usr/bin/env python
"""
model
=====
Defines BaseModel class used for internal simulations.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import master_util
import numpy as np
import re

class BaseModel:
    def __init__(self, args):
        """
        Initializes the BaseModel.

        Args:
            args (dict): A dictionary containing the arguments for initialization.
        """
        return
    
    def set_args(self, args):
        """
        Sets the arguments for the model.

        Args:
            args (dict): A dictionary containing the arguments.
        """
        self.model_type         = args['model_type']
        self.model_variant      = args['model_variant']
        self.rv_fn              = args['rv_fn']
        self.rv_arg             = args['rv_arg']
        self.sim_dir            = args['sim_dir']
        self.proj               = args['proj']
        self.min_num_taxa       = args['min_num_taxa']
        self.max_num_taxa       = args['max_num_taxa']
        self.sample_population  = args['sample_population']
        self.stop_time          = args['stop_time']
        return
    
    def set_model(self, seed=None):
        """
        Sets the model.

        Args:
            seed (int, optional): The random seed value. Defaults to None.
        """
        # set RNG seed if provided
        
        #print("BaseModel.set_model", seed)
        #np.random.seed(seed=seed)
        # set RNG
        self.seed        = seed
        self.rng         = np.random.Generator(np.random.PCG64(seed))
        # state space
        self.states      = self.make_states() # self.num_locations )
        # params space
        self.params      = self.make_params()
        # starting population sizes (e.g. SIR models)
        # self.start_sizes = self.make_start_sizes()
        # starting state
        self.start_state, self.start_sizes = self.make_start_conditions()
        # event space
        self.events      = self.make_events( self.states, self.params )
        # event space dataframe
        self.df_events   = master_util.events2df( self.events )
        # state space dataframe
        self.df_states   = master_util.states2df( self.states )
        # reaction vars
        self.reaction_vars = self.make_reaction_vars()
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
        newick_fn = '{sim_dir}/{proj}/sim.{idx}.tre'.format(sim_dir=self.sim_dir, proj=self.proj, idx=idx)
        nexus_fn  = '{sim_dir}/{proj}/sim.{idx}.phy.nex'.format(sim_dir=self.sim_dir, proj=self.proj, idx=idx)
        json_fn   = '{sim_dir}/{proj}/sim.{idx}.json'.format(sim_dir=self.sim_dir, proj=self.proj, idx=idx)

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
    
    def clear_model(self):
        """
        Clears the model.
        """
        self.is_model_set = False
        self.states = None
        self.params = None
        self.events = None
        self.df_events = None
        self.df_states = None
        return
    
    def make_settings(self):
        """
        Creates the settings for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_states(self):
        """
        Creates the state space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_events(self):
        """
        Creates the event space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_params(self):
        """
        Creates the parameter space for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_start_state(self):
        """
        Creates the starting state for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
    def make_start_sizes(self):
        """
        Creates the starting sizes for the model.

        Raises:
            NotImplementedError: This method should be implemented in derived classes.
        """
        raise NotImplementedError
    
