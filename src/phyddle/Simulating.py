#!/usr/bin/env python
"""
Simulating
==========
Defines classes and methods for the Simulating step, which generates large numbers
of simulated datasets (in parallel, if desired) that are later formatted and used
to train the neural network.

Author:    Michael Landis
Copyright: (c) 2023, Michael Landis
License:   MIT
"""

# standard imports
import gzip
import os
import re
import shutil
import subprocess

# external imports
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# phyddle imports
from phyddle import Utilities

#-----------------------------------------------------------------------------------------------------------------#

class Simulator:
    def __init__(self, args, mdl):
        self.set_args(args)
        self.command      = 'echo \"phyddle.Simulator.command undefined in derived class!\"' # do nothing
        self.model        = mdl
        return

    def set_args(self, args):
        # simulator arguments
        self.args              = args
        self.proj              = args['proj']
        self.sim_dir           = args['sim_dir']
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
        return

    def make_settings_str(self, idx, mtx_size):

        s = 'setting,value\n'
        s += 'proj,' + self.proj + '\n'
        s += 'model_name,' + self.model.model_type + '\n'
        s += 'model_variant,' + self.model.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'taxon_category,' + str(mtx_size) + '\n'
        return s

    def run(self):
        # prepare workspace
        os.makedirs( self.sim_dir + '/' + self.proj, exist_ok=True )
        # dispatch jobs
        if self.use_parallel:
            res = Parallel(n_jobs=self.num_proc)(delayed(self.sim_one)(idx) for idx in tqdm(self.rep_idx))
        else:
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx) ]
        return res

    # main simulation function (looped)
    def sim_one(self, idx):
        
        # improve numpy output style (move to Utilities??)
        NUM_DIGITS = 10
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')}, precision=NUM_DIGITS)
        
        # make filenames
        out_path  = self.sim_dir + '/' + self.proj + '/sim'
        tmp_fn    = out_path + '.' + str(idx)
        param_mtx_fn = tmp_fn + '.param_col.csv'
        param_vec_fn = tmp_fn + '.param_row.csv'
        
        # refresh model and update XML string (redraw parameters, etc.)
        self.refresh_model(idx)
        
        # record labels (simulating parameters)
        param_mtx_str,param_vec_str = Utilities.param_dict_to_str(self.model.params)
        Utilities.write_to_file(param_mtx_str, param_mtx_fn)
        Utilities.write_to_file(param_vec_str, param_vec_fn)
        
        # delegate simulation to derived Simulator
        self.sim_one_custom(idx)

        # done!
        return

    def refresh_model(self, idx):
        self.model.set_model(idx)
        self.start_state   = self.model.start_state
        self.start_sizes   = self.model.start_sizes
        self.df_events     = self.model.df_events
        self.refresh_model_custom(idx)
        return

    def sim_one_custom(self, idx):
        return NotImplementedError
    
    def refresh_model_custom(self, idx):
        return NotImplementedError    

#-----------------------------------------------------------------------------------------------------------------#

###################################
# Generic CLI simulator interface #
###################################

# must be command line tool
# - accepts input model parameters
# - outputs Newick tree file
# - outputs Nexus data matrix
# - phyddle handles rest

class GenericSimulator(Simulator):
    def __init__(self, args, mdl):
        super().__init__(args, mdl)
        return


#-----------------------------------------------------------------------------------------------------------------#

##############################
# MASTER simulator interface #
##############################

class MasterSimulator(Simulator):
    def __init__(self, args, mdl):
        # call base constructor
        super().__init__(args, mdl)
        return
    
    # def refresh_model(self, idx):
    #     self.model.set_model(idx)
    #     self.start_state   = self.model.start_state
    #     self.start_sizes   = self.model.start_sizes
    #     self.df_events     = self.model.df_events
    #     self.reaction_vars = self.make_reaction_vars()
    #     self.xml_str       = self.make_xml(idx)
    #     self.cmd_str       = 'beast {sim_dir}/{proj}/sim.{idx}.xml'.format(sim_dir=self.sim_dir, proj=self.proj, idx=idx)
    #     return

    def refresh_model_custom(self, idx):
        self.reaction_vars = self.make_reaction_vars()
        self.xml_str       = self.make_xml(idx)
        self.cmd_str       = 'beast {sim_dir}/{proj}/sim.{idx}.xml'.format(sim_dir=self.sim_dir, proj=self.proj, idx=idx)


    def sim_one_custom(self, idx):
        out_path   = self.sim_dir + '/' + self.proj + '/sim'
        tmp_fn     = out_path + '.' + str(idx)

        beast_fn   = tmp_fn + '.beast.log'
        xml_fn     = tmp_fn + '.xml'
        json_fn    = tmp_fn + '.json'
        phy_nex_fn = tmp_fn + '.phy.nex' # annotated tree, no mtx
        dat_nex_fn = tmp_fn + '.dat.nex' # no tree, character mtx
        tre_fn     = tmp_fn + '.tre'

        # make XML file
        xml_str = self.xml_str
        Utilities.write_to_file(xml_str, xml_fn)

        # run BEAST job
        cmd_str = self.cmd_str
        beast_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
        Utilities.write_to_file(beast_out, beast_fn)

        # this code should convert to standard nexus
        # convert MASTER tree to standard nexus
        #taxon_states,
        # state space
        int2vec = self.model.states.int2vec
        nexus_str = Utilities.convert_phy2dat_nex(phy_nex_fn, int2vec)
        Utilities.write_to_file(nexus_str, dat_nex_fn)


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
        #return xml_spec_str
