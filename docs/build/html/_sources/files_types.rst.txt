Files & datatypes
========================

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

Phylogenetic data (e.g. from a Newick file) and character matrix data (e.g. from a Nexus file) are encoded into compact phylogenetic state tensors. Internally, phyddle uses `dendropy.Tree` to represent phylogenies, `pandas.DataFrame` to represent character matrices (verify), and `numpy.ndarray` to store phylogenetic-state tensors.

There are two types of phylogenetic-state tensors in phyddle: the compact bijective ladderized vector + states (CBLV+S) and the compact diversity vector + states (CDV+S). CBLV+S is used for trees that contain serially sampled (non-ultrametric) taxa whereas CDV+S is used for trees that contain only extant (ultrametric) taxa. The `tree_width` of the encoding defines the maximum number of taxa the phylogenetic-state tensor may contain.

CBLV+S
^^^^^^

This is an example for the CBLV+S encoding of 5 taxa with 2 characters. This is the Newick string:

.. code-block:: shell

    (((A:2,B:1):1,(C:3,D:2):3):1,E:2);


This is the Nexus file:

.. code-block:: shell

    #NEXUS
    Begin DATA;
    Dimensions NTAX=5 NCHAR=2
    Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
    Matrix
        A  01
        B  11
        C  10
        D  10
        E  01
    ;
    END;


Ladderizing clades by maximum root-to-tip distance orders the taxa C, D, A, B, then E, which correspond to the first five columns of the CBLV+S tensor.
The un-rescaled CBLV+S file would look like this:

.. code-block:: shell

    # C,D,A,B,E,-,-,-,-,-  
      3,2,2,1,2,0,0,0,0,0  # tip edge length
      0,3,1,1,0,0,0,0,0,0  # node edge length
      7,2,3,1,2,0,0,0,0,0  # tip-to-node distance
      0,4,1,2,0,0,0,0,0,0  # node-to-root distance
      1,1,0,1,0,0,0,0,0,0  # character 1
      0,0,1,1,1,0,0,0,0,0  # character 2


By default, all branch length entries are rescaled from 0 to 1 as proportion to tree height (formatted to ease reading):

.. code-block:: shell

    #    C,   D,   A,   B,   E,   -,   -,   -,   -,   -  
      0.43,0.29,0.29,0.14,0.29,   0,   0,   0,   0,   0  # tip edge length
      0.00,0.43,0.14,0.14,0.00,   0,   0,   0,   0,   0  # node edge length
      1.00,0.29,0.43,0.14,0.29,   0,   0,   0,   0,   0  # tip-to-node distance
      0.00,0.57,0.14,0.29,0.00,   0,   0,   0,   0,   0  # node-to-root distance
         1,   1,   0,   1,   0,   0,   0,   0,   0,   0  # character 1
         0,   0,   1,   1,   1,   0,   0,   0,   0,   0  # character 2



Output files
------------


The results file contains the predictions made by phyddle's neural network for a new dataset:

.. code-block:: shell

   $ cat new.1.sim_batchsize128_numepoch20_nt500.pred_labels.csv
   w_0_value,w_0_lower,w_0_upper,e_0_value,e_0_lower,e_0_upper,d_0_1_value,d_0_1_lower,d_0_1_upper,b_0_1_value,b_0_1_lower,b_0_1_upper
   0.2867125345651129,0.1937433853918723,0.45733220552078013,0.02445545359384659,0.002880695707341881,0.10404499205878459,0.4502031713887769,0.1966340488593367,0.5147956690178682,0.06199703190510973,0.0015074254823161301,0.27544015163806645

Columns are grouped first by label (e.g. parameter) and then statistic (e.g. value, lower-bound, upper-bound).


