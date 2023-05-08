import os
import re
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm

# look into reorganizing this stuff, removing unneeded files
from phyddle_util import *
import cdvs_util

# needed?
#import model
#import time

class Simulator:
    def __init__(self, args, mdl): #, events_df, states_df):
        self.set_args(args)
        self.command      = 'echo \"phyddle.Simulator.command undefined in derived class!\"' # do nothing
        self.model        = mdl
        return

    def set_args(self, args):
        # simulator arguments
        self.args              = args
        self.job_name          = args['job_name']
        self.sim_dir           = args['sim_dir']
        self.start_idx         = args['start_idx']
        self.end_idx           = args['end_idx']
        self.tree_sizes        = args['tree_sizes']
        self.num_proc          = args['num_proc']
        self.use_parallel      = args['use_parallel']
        self.sample_population = args['sample_population']
        self.stop_floor_sizes  = args['stop_floor_sizes']
        self.stop_ceil_sizes   = args['stop_ceil_sizes']
        self.start_sizes       = args['start_sizes']
        self.start_state       = args['start_state']
        self.rep_idx           = list(range(self.start_idx, self.end_idx))
        return

    def make_settings_str(self, idx, mtx_size):

        s = 'setting,value\n'
        s += 'job_name,' + self.job_name + '\n'
        s += 'model_name,' + self.model.model_type + '\n'
        s += 'model_variant,' + self.model.model_variant + '\n'
        s += 'replicate_index,' + str(idx) + '\n'
        s += 'taxon_category,' + str(mtx_size) + '\n'
        return s

    def run(self):
        # prepare workspace
        os.makedirs( self.sim_dir + '/' + self.job_name, exist_ok=True )
        # dispatch jobs
        if self.use_parallel:
            res = Parallel(n_jobs=self.num_proc)(delayed(self.sim_one)(idx) for idx in tqdm(self.rep_idx))
        else:
            res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx) ]
        return res

    # main simulation function (looped)
    def sim_one(self, idx):
        
        # make filenames
        out_path  = self.sim_dir + '/' + self.job_name + '/sim'
        tmp_fn    = out_path + '.' + str(idx)
        geo_fn    = tmp_fn + '.geosse.nex'
        tre_fn    = tmp_fn + '.tre'
        prune_fn  = tmp_fn + '.extant.tre'
        beast_fn  = tmp_fn + '.beast.log'
        xml_fn    = tmp_fn + '.xml'
        nex_fn    = tmp_fn + '.nex'
        #json_fn   = tmp_fn + '.json'
        cblvs_fn  = tmp_fn + '.cblvs.csv'
        cdvs_fn   = tmp_fn + '.cdvs.csv'
        param1_fn = tmp_fn + '.param1.csv'
        param2_fn = tmp_fn + '.param2.csv'
        ss_fn     = tmp_fn + '.summ_stat.csv'
        info_fn   = tmp_fn + '.info.csv'

        # refresh model and update XML string (redraw parameters, etc.)
        self.refresh_model(idx)
        int2vec = self.model.states.int2vec
        int2vecstr = self.model.states.int2vecstr #[ ''.join([str(y) for y in x]) for x in int2vec ]
        vecstr2int = self.model.states.vecstr2int #{ v:i for i,v in enumerate(int2vecstr) }

        # make XML file
        xml_str = self.xml_str
        write_to_file(xml_str, xml_fn)

        # run BEAST job
        cmd_str = self.cmd_str
        beast_out = subprocess.check_output(cmd_str, shell=True, text=True, stderr=subprocess.STDOUT)
        write_to_file(beast_out, beast_fn)

        # verify tree size & existence!
        result_str     = ''
        n_taxa_idx     = get_num_taxa(tre_fn, idx, self.tree_sizes)
        taxon_size_idx = find_taxon_size(n_taxa_idx, self.tree_sizes)

        # handle simulation based on tree size
        if n_taxa_idx > np.max(self.tree_sizes):
            # too many taxa
            result_str = '- replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)
            return result_str
        elif n_taxa_idx <= 0:
            # no taxa
            result_str = '- replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)
            return result_str
        else:
            # valid number of taxa
            result_str = '+ replicate {idx} simulated n_taxa={nt}'.format(idx=idx, nt=n_taxa_idx)

            # generate extinct-pruned tree
            prune_success = make_prune_phy(tre_fn, prune_fn)

            # MJL 230411: probably too aggressive, should revisit
            if not prune_success:
                next

            # generate nexus file 0/1 ranges
            taxon_states,nexus_str = convert_nex(nex_fn, tre_fn, int2vec)
            write_to_file(nexus_str, geo_fn)

            # then get CBLVS working
            cblv,new_order = vectorize_tree(tre_fn, max_taxa=taxon_size_idx, prob=1.0 )
            cblvs = make_cblvs_geosse(cblv, taxon_states, new_order)
        
            # NOTE: this if statement should not be needed, but for some reason the "next"
            # seems to run even when make_prune_phy returns False
            # generate CDVS file
            if prune_success:
                cdvs = cdvs_util.make_cdvs(prune_fn, taxon_size_idx, taxon_states, int2vecstr)

            # output files
            mtx_size = cblv.shape[1]

        # record info
        info_str = self.make_settings_str(idx, mtx_size)
        write_to_file(info_str, info_fn)

        # record labels (simulating parameters)
        param1_str,param2_str = param_dict_to_str(self.model.rates)
        write_to_file(param1_str, param1_fn)
        write_to_file(param2_str, param2_fn)

        # record CBLVS data
        cblvs_str = np.array2string(cblvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200)
        cblvs_str = cblvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
        write_to_file(cblvs_str, cblvs_fn)

        # record CDVS data
        if prune_success:
            cdvs = cdvs.to_numpy()
            cdvs_str = np.array2string(cdvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200)
            cdvs_str = cdvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
            write_to_file(cdvs_str, cdvs_fn)

        # record summ stat data
        ss = make_summ_stat(tre_fn, geo_fn, vecstr2int)
        ss_str = make_summ_stat_str(ss)
        write_to_file(ss_str, ss_fn)

        # return status string
        return result_str

class MasterSimulator(Simulator):
    def __init__(self, args, mdl):
        # call base constructor
        super().__init__(args, mdl)
        return
    
    def refresh_model(self, idx):
        self.model.set_model(idx)
        self.df_events     = self.model.df_events
        self.reaction_vars = self.make_reaction_vars()
        self.xml_str       = self.make_xml(idx)
        self.cmd_str       = 'beast {sim_dir}/{job_name}/sim.{idx}.xml'.format(sim_dir=self.sim_dir, job_name=self.job_name, idx=idx)
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
        newick_fn = '{sim_dir}/{job_name}/sim.{idx}.tre'.format(sim_dir=self.sim_dir, job_name=self.job_name, idx=idx)
        nexus_fn  = '{sim_dir}/{job_name}/sim.{idx}.nex'.format(sim_dir=self.sim_dir, job_name=self.job_name, idx=idx)
        json_fn   = '{sim_dir}/{job_name}/sim.{idx}.json'.format(sim_dir=self.sim_dir, job_name=self.job_name, idx=idx)

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
                xml_init_state += "\t<populationSize spec='PopulationSize' size='{y}'>\n".format(y=y)
                xml_init_state += "\t\t<population spec='Population' type='@{k}' location='{i}'/>\n".format(k=k,i=i)
                xml_init_state += "\t</populationSize>\n"
        for k,v in self.start_state.items():
            xml_init_state += "\t<lineageSeedMultiple spec='MultipleIndividuals' copies='1'>\n".format(v=v)
            xml_init_state += "\t\t<population spec ='Population' type='@{k}' location='{v}'/>\n".format(k=k,v=v)
            xml_init_state += "\t</lineageSeedMultiple>\n"
        xml_init_state += "</initialState>\n"

        # sim conditions
        xml_sim_conditions = ""
        xml_sim_conditions += "<lineageEndCondition spec='LineageEndCondition' nLineages='{stop_ceil_sizes}' alsoGreaterThan='true' isRejection='false'/>\n".format(stop_ceil_sizes=self.stop_ceil_sizes)
        xml_sim_conditions += "<lineageEndCondition spec='LineageEndCondition' nLineages='{stop_floor_sizes}' alsoGreaterThan='false' isRejection='false'/>\n".format(stop_floor_sizes=self.stop_floor_sizes)
        xml_sim_conditions += "<postSimCondition spec='LeafCountPostSimCondition' nLeaves='10' exact='false' exceedCondition='true'/>\n"

        # post-processing filter
        sample_population = self.sample_population #settings['sample_population']
        xml_filter = ''
        for k in sample_population:
            xml_filter += "<inheritancePostProcessor spec='LineageFilter' populationName='{k}' reverseTime='false' discard='false' leavesOnly='true' noClean='true'/>\n".format(k=k)

        # generate entire XML specification
        xml_spec_str = '''\
<beast version='2.0' namespace='master:master.model:master.steppers:master.conditions:master.postprocessors:master.outputs'>

<run spec='InheritanceEnsemble'
    verbosity='3'
    nTraj='1'
    nSamples='{num_samples}'
    samplePopulationSizes='{sample_pop}'
    simulationTime='10'
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
           num_samples=1,
           sample_pop='false')
        
        return xml_spec_str
        #return xml_spec_str

