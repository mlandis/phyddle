import sys
import math
import os
import subprocess
from natsort import natsorted
import pandas as pd

__author__ = "Fabio K. Mendes"
__email__ = "f.mendes@wustl.edu"


# from scripts/
# $ sudo python3 sim/PhyloJunction/sim_one.py sim/PhyloJunction/bisse_timehet.pj trs sim/PhyloJunction/ 1 10

def parse_PJ_tree_tsv(out_path,
                      prefix,
                      tree_tsv_path,
                      tree_state_csv_paths,
                      tree_node_name,
                      idx):

    rec_tr_df = pd.read_csv(tree_tsv_path, sep='\t', header=0)

    if len (set(rec_tr_df['replicate'])) >= 2:
        exit("Detected more than one tree replicate per simulation. Exiting.")

    tr_list = rec_tr_df[tree_node_name].tolist()
    n_trs = len(tr_list)
    for i in range(n_trs):
        j = i + 1

        # write parsed tree file
        parsed_tr_path = prefix + "." + str(idx + i) + ".tre"
        with open(out_path + parsed_tr_path, "w") as outfile:
            print(tr_list[i], file=outfile)

        # write states to csv file
        parsed_states_path = prefix + "." + str(idx + i) + ".dat.csv"
        with open(out_path + parsed_states_path, "w") as outfile:
            with open(tree_state_csv_paths[i]) as infile:
                lines = infile.readlines()
                lines[-1] = lines[-1].rstrip() # remove the last new line character
                content = "taxa,data\n" + "".join(l.replace("\t", ",") for l in lines) # add header
                print(content, file=outfile)


def parse_PJ_scalar_tsv(out_path, prefix, scalar_csv_path, idx):

    # write scalar csv file
    with open(scalar_csv_path, "r") as infile:
        header = infile.readline().split(",")
        header = [ 'log10_' + x for x in header ]
        rv_names = ",".join(header[2:]) # skip sim and repl

        for i, line in enumerate(infile):
            vals = [ math.log(float(x), 10) for x in line.split(',')[2:] ]
            vals = ','.join([str(x) for x in vals]).rstrip()

            with open(out_path + prefix + "." + str(idx + i) + ".labels.csv", "w") as outfile:
                print(rv_names + vals, file=outfile)


if __name__ == "__main__":

    ###########################
    # get and parse arguments #
    ###########################
   
    pj_script_path = sys.argv[1]
    if not os.path.isfile(pj_script_path):
        exit("Could not find " + pj_script_path + ". Exiting.")

    # e.g. trs_1
    tree_node_name = sys.argv[2]

    pj_out_path = sys.argv[3]
    if not pj_out_path.endswith("/"):
        pj_out_path += "/"

    prefix = sys.argv[4]
    idx = int(sys.argv[5]) # used to name files
    sim_prefix = prefix + '.' + str(idx)

    n_batches = sys.argv[6] # PJ's number of samples

    ##################
    # preparing dirs #
    ##################

    # output of this script
    out_path = pj_out_path # + "parsed_sim_output/"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        print("Created", out_path)

    #####################
    # editing PJ script #
    #####################

    head, tail = os.path.split(pj_script_path)
    if head == '':
        head = '.'
    if not head.endswith("/"):
        head += "/"
        
    parsed_pj_script_path = head + tail.replace(".pj", "_" + str(idx) + "_parsed.pj")
    with open(parsed_pj_script_path, "w") as outfile:
        with open(pj_script_path, "r") as infile:
            for line in infile:
                line = line.rstrip()

                if line.startswith("n_batches <-"):
                    print("n_batches <- " + n_batches, file=outfile)
                #elif line.startswith("trs ~"):
                #    line = line.replace("trs ~", tree_node_name + " ~")
                #    print(line, file=outfile)
                else:
                    print(line, file=outfile)

    print("Successfully updated PhyloJunction script (in", parsed_pj_script_path + ")")
            
    ###########
    # call PJ #
    ###########

    call_pj = True
    if call_pj:
        pj_args = ["pjcli", parsed_pj_script_path, "-d", "-r", str(idx), "-o", pj_out_path, "-p", sim_prefix ]
        p = subprocess.Popen(pj_args, stdout=subprocess.PIPE)
        pj_bytes = p.communicate()[0]
        pj_msg = pj_bytes.decode('utf-8')
        print("PhyloJunction log:\n", pj_msg)

    #####################
    # reading PJ output #
    #####################

    # get tree tsv path
    tree_tsv_path = pj_out_path + sim_prefix + "_" + tree_node_name + "_reconstructed.tsv"
    if not os.path.isfile(tree_tsv_path):
        exit("Could not find " + tree_tsv_path + ". Exiting.")

    # get tree state csv paths
    tree_state_csv_paths = natsorted([pj_out_path + f for f in os.listdir(pj_out_path) \
                            if f.endswith("repl1.tsv") and f.startswith(sim_prefix + "_" + tree_node_name + "_")])
    if len(tree_state_csv_paths) == 0:
        exit("Could not find any .tsv file containing tip states. Exiting.")

    # get scalar csv path
    scalar_csv_path = pj_out_path + sim_prefix + "_scalar_rvs_repl1.csv"
    if not os.path.isfile(scalar_csv_path):
        exit("Could not find " + scalar_csv_path + ". Exiting.")
    
    #####################
    # parsing PJ output #
    #####################

    parse_PJ_tree_tsv(out_path,
                      prefix,
                      tree_tsv_path,
                      tree_state_csv_paths,
                      tree_node_name,
                      idx)

    parse_PJ_scalar_tsv(out_path,
                        prefix,
                        scalar_csv_path,
                        idx)


    ###########
    # cleanup #
    ###########
    
    os.remove(parsed_pj_script_path)
