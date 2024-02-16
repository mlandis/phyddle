import sys
import os
import subprocess
from natsort import natsorted
import pandas as pd

__author__ = "Fabio K. Mendes"
__email__ = "f.mendes@wustl.edu"


# from scripts/
# $ sudo python3 sim/PhyloJunction/sim_one.py sim/PhyloJunction/ 1 sim/PhyloJunction/bisse_timehet.pj trs

def parse_PJ_tree_tsv(out_path,
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
        parsed_tr_path = "sim." + str(idx + i) + ".tre"
        with open(out_path + parsed_tr_path, "w") as outfile:
            print(tr_list[i], file=outfile)

        # write states to csv file
        parsed_states_path = "sim." + str(idx + i) + ".dat.csv"
        with open(out_path + parsed_states_path, "w") as outfile:
            with open(tree_state_csv_paths[i]) as infile:
                lines = infile.readlines()
                lines[-1] = lines[-1].rstrip() # remove the last new line character
                content = "taxa,data\n" + "".join(l.replace("\t", ",") for l in lines) # add header
                print(content, file=outfile)


def parse_PJ_scalar_tsv(out_path, scalar_csv_path, idx):

    # write scalar csv file
    with open(scalar_csv_path, "r") as infile:
        header = infile.readline().split(",")
        rv_names = ",".join(header[2:]) # skip sim and repl
        
        for i, line in enumerate(infile):
            vals = ",".join(line.split(",")[2:]).rstrip()

            with open(out_path + "sim." + str(idx + i) + ".labels.csv", "w") as outfile:
                print(rv_names + vals, file=outfile)


if __name__ == "__main__":

    ###########################
    # get and parse arguments #
    ###########################
    pj_out_path = sys.argv[1]
    if not pj_out_path.endswith("/"):
        pj_out_path += "/"

    idx = int(sys.argv[2]) # used to name files

    pj_script_path = sys.argv[3]
    if not os.path.isfile(pj_script_path):
        exit("Could not find " + pj_script_path + ". Exiting.")

    tree_node_name = sys.argv[4]

    # output of this script
    out_path = pj_out_path + "parsed_sim_output/"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        print("Created", out_path)

    ###########
    # call PJ #
    ###########

    call_pj = True
    if call_pj:
        pj_args = ["pjcli", pj_script_path, "-d", "-o", pj_out_path]
        p = subprocess.Popen(pj_args, stdout=subprocess.PIPE)
        pj_bytes = p.communicate()[0]
        pj_msg = pj_bytes.decode('utf-8')
        print("PhyloJunction log:\n", pj_msg)

    #####################
    # reading PJ output #
    #####################

    # get tree tsv path
    tree_tsv_path = pj_out_path + tree_node_name + "_reconstructed.tsv"
    if not os.path.isfile(tree_tsv_path):
        exit("Could not find " + tree_tsv_path + ". Exiting.")

    # get tree state csv paths
    tree_state_csv_paths = natsorted([pj_out_path + f for f in os.listdir(pj_out_path) \
                            if f.endswith("repl1.tsv") and f.startswith(tree_node_name)])
    if len(tree_state_csv_paths) == 0:
        exit("Could not find any .tsv file containing tip states. Exiting.")

    # get scalar csv path
    scalar_csv_path = pj_out_path + "scalar_rvs_repl1.csv"
    if not os.path.isfile(scalar_csv_path):
        exit("Could not find " + scalar_csv_path + ". Exiting.")
    
    #####################
    # parsing PJ output #
    #####################

    parse_PJ_tree_tsv(out_path,
                      tree_tsv_path,
                      tree_state_csv_paths,
                      tree_node_name,
                      idx)

    parse_PJ_scalar_tsv(out_path,
                        scalar_csv_path,
                        idx)
