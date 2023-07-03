File formats
============

.. note::

    Incomplete.


.. _fmt_input_files:

Input files
-----------

Trees are encoded as raw data in simple Newick format. Trees are assumed to be rooted, bifurcating, time-calibrated trees. Trees may be ultrametric or non-ultrametric trees. Ultrametric trees should only be analyzed using `treetype == 'extant'`. Non-ultrametric trees, such as those containing serially sampled viruses or fossils should be analyzed using `treetype == 'serial'`. Here is an example of an extant tree with N=8 taxa.

.. code-block:: shell
   
    $ cat workspace/raw_data/example/sim.3.tre
    ((((1:0.35994691486501296,2:0.35994691486501296):1.389952711060852,(3:1.5810568349100933,(4:0.5830569936279364,5:0.5830569936279364):0.9979998412821569):0.1688427910157717):5.655066077200624,6:7.404965703126489):0.3108578683347094,(7:0.7564319839861859,8:0.7564319839861859):6.959391587475013):2.2841764285388018;


Character data matrices are encoded as raw data in Nexus format. Here is an example of a matrix with N=8 taxa and M=3 binary characters.

.. code-block:: shell

    $ cat workspace/raw_data/example/sim.3.dat.nex
    #NEXUS
    Begin DATA;
    Dimensions NTAX=8 NCHAR=3
    Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
    Matrix
        1  001
        2  010
        3  100
        4  100
        5  001
        6  001
        7  100
        8  010
    ;
    END;


Some models can make use of auxiliary data. For example, the recovery period of a virus may be considered a "known" parameter in epidemiological studies. These auxiliary data are encoded as raw data in comma-separated value format with column headers.

.. code-block:: shell

   recovery_0,recovery_1,recovery_2
   0.113,0.120,0.115


Tensor files
------------

Phylogenetic data (e.g. from a Newick file) and character matrix data (e.g. from a Nexus file) are encoded into compact phylogenetic state tensors.



Compact bijective ladderized vector + states

.. code-block:: shell

    1,1,2,3,1,3,3,0,0,0,0
    1,2,3,2,3,2,0,0,0,0,0
    1,1,1,1,1,1,1,1,1,1,1

Compact diversity vector + states

.. code-block:: shell

    1,1,2,3,1,3,3,0,0,0,0
    1,1,1,1,1,1,1,1,1,1,1


Output files
------------


The results file contains the predictions made by phyddle's neural network for a new dataset:

.. code-block:: shell

   $ cat new.1.sim_batchsize128_numepoch20_nt500.pred_labels.csv
   w_0_value,w_0_lower,w_0_upper,e_0_value,e_0_lower,e_0_upper,d_0_1_value,d_0_1_lower,d_0_1_upper,b_0_1_value,b_0_1_lower,b_0_1_upper
   0.2867125345651129,0.1937433853918723,0.45733220552078013,0.02445545359384659,0.002880695707341881,0.10404499205878459,0.4502031713887769,0.1966340488593367,0.5147956690178682,0.06199703190510973,0.0015074254823161301,0.27544015163806645

Columns are grouped first by label (e.g. parameter) and then statistic (e.g. value, lower-bound, upper-bound).


