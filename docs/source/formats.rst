.. _Formats:

Formats
=======

.. note::

    (incomplete) Important: This section assume the project name is 'example' while actual projects will likely use different names. Visit :ref:`Glossary` to learn more about
    how phyddle defines different terms.

This page describes different internal datatype formats and file formats used
by phyddle.

.. _fmt_input_files:

Input datasets
--------------

phyddle can make phylogenetic model predictions against input datasets with
previously trained networks. Valid phyddle input datasets contain a set of
files with a shared filename prefix. For example, a dataset with the prefix
``sim.3`` would contain a tree file ``sim.3.tre``, a character matrix file
``sim.3.dat.nex``, and (when applicable) a 'known parameters' file
``sim.3.known_params.csv``. Simulated training datasets and real biological
datasets follow the same format.

Trees are encoded as raw data in simple Newick format. Trees are assumed to be
rooted, bifurcating, time-calibrated trees. Trees may be ultrametric or
non-ultrametric trees. Ultrametric trees should only be analyzed using
`treetype == 'extant'`. Non-ultrametric trees, such as those containing
serially sampled viruses or fossils should be analyzed using
`treetype == 'serial'`. Here is an example of an extant tree with N=8 taxa.

.. code-block:: shell
   
    $ cat workspace/raw_data/example/sim.3.tre
    ((((1:0.35994691486501296,2:0.35994691486501296):1.389952711060852,(3:1.5810568349100933,(4:0.5830569936279364,5:0.5830569936279364):0.9979998412821569):0.1688427910157717):5.655066077200624,6:7.404965703126489):0.3108578683347094,(7:0.7564319839861859,8:0.7564319839861859):6.959391587475013):2.2841764285388018;


Character data matrices are encoded as raw data in Nexus format. Here is an
example of a matrix with N=8 taxa and M=3 binary characters.

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


Some models can make use of auxiliary data. For example, the recovery period
of a virus may be considered a "known" parameter in epidemiological studies.
These auxiliary data are encoded as raw data in comma-separated value format
with column headers.

.. code-block:: shell

   recovery_0,recovery_1,recovery_2
   0.113,0.120,0.115

.. _Tensor_Formats:

Tensor formats
--------------

Phylogenetic data (e.g. from a Newick file) and character matrix data (e.g.
from a Nexus file) are encoded into compact phylogenetic state tensors.
Internally, phyddle uses `dendropy.Tree` to represent phylogenies,
`pandas.DataFrame` to represent character matrices (verify), and
`numpy.ndarray` to store phylogenetic-state tensors.

..
    CBLV encodes a phylogenetic tree with $n \leq N$ taxa in to a matrix of with 2 rows that contains branch length sorted across $N$ columns that contain topological information for a tree with taxa serially sampled over time (e.g. epidemiological data). The matrix is then flattened into vector format. Ammon et al. (2022) introduced the CBLV+S format, which allows for multiple characters to be associated with each taxon in a CBLV, constructing a matrix with $2+M$ rows and $N$ columns for a dataset of $n \leq N$ taxa with $M$ characters. Another important tensor type developed by Lambert et al. (2022) is the compact diversified vector (CDV). CDV is a matrix with 2 rows and $N$ columns, with the first row corresponding to node ages and the other recording state values for a single binary character.

    CBLV and CDV differ primarily in terms what criteria they use to they order (ladderize) the topology. CBLV ladderizes by minimum terminal-node age per clade and CDV ladderized by maximum subclade branch length. Both formats pack the phylogenetic information from a tree with $n$ taxa into a "wider" tree-width class that allows up to $N$ taxa. The tensor is packed from left-to-right based on an in-order tree traversal, then use zeroes to buffer the all remaining cells until the $N$th column. In phyddle, we use expanded CBLV+S and CDV+S formats that additionally encode terminal branch length formation for the terminal node and the parent node, resulting in $4+M$ rows for our CBLV+S and $3+M$ rows for our CDV+S format. (Will add diagram later.)

    The second input is the **auxiliary data tensor**. This tensor contains summary statistics for the phylogeny and character data matrix and "known" parameters for the data generating process. The summary statistics, for example, report things such as the number of taxa, the tree height, the mean and variance of branch lengths and node ages, the state-pattern counts, etc. The known parameters might report things such as the population sizes of a susceptible population or the recovery period in an SIR model.

There are two types of phylogenetic-state tensors in phyddle: the compact
bijective ladderized vector + states (CBLV+S) and the compact diversity vector +
states (CDV+S). CBLV+S is used for trees that contain serially sampled
(non-ultrametric) taxa whereas CDV+S is used for trees that contain only extant
(ultrametric) taxa. The `tree_width` of the encoding defines the maximum number
of taxa the phylogenetic-state tensor may contain. The ``tree_type`` setting
determines if the tree is a ``'serial'`` tree encoded with CBLV+S or an
``'extant'`` tree encoded with CDV+S. Setting ``tree_encode_type`` and
``char_encode_type`` alter how information is stored into the
phylogenetic-state tensor.

CBLV+S
^^^^^^

This is an example for the CBLV+S encoding of 5 taxa with 2 characters. This
is the Newick string:

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


These data can be encoded in different ways, based on the ``char_encode_type``
setting. When ``char_encode_type == 'integer'`` then the encoding will treat
each character as a row in the resulting data matrix, and assign the
appropriate integer-valued state to that character for each taxon.
Alternatively, when ``char_encode_type == 'one_hot'`` then the encoding will
treat every distinct state-character combination as its own row in the
resulting data matrix, then mark each species as ``1`` for a cell when a
species has that character-state and ``0`` if not. One-hot encoding is
applied individually to each homologous character (fewer distinct combinations)
not against the entire character set (more distinct combinations).


Ladderizing clades by maximum root-to-tip distance orders the taxa C, D, A,
B, then E, which correspond to the first five columns of the CBLV+S tensor.
When ``tree_encode_type`` is set to ``'height_only'`` the un-rescaled CBLV+S file
would look like this:

.. code-block:: shell

    # C,D,A,B,E,-,-,-,-,-  
      7,2,3,1,2,0,0,0,0,0  # tip-to-node distance
      0,4,1,2,0,0,0,0,0,0  # node-to-root distance
      1,1,0,1,0,0,0,0,0,0  # character 1
      0,0,1,1,1,0,0,0,0,0  # character 2


and like this when ``tree_encode_type`` is set to ``'height_brlen'``:

.. code-block:: shell

    # C,D,A,B,E,-,-,-,-,-  
      7,2,3,1,2,0,0,0,0,0  # tip-to-node distance
      0,4,1,2,0,0,0,0,0,0  # node-to-root distance
      3,2,2,1,2,0,0,0,0,0  # tip edge length
      0,3,1,1,0,0,0,0,0,0  # node edge length
      1,1,0,1,0,0,0,0,0,0  # character 1
      0,0,1,1,1,0,0,0,0,0  # character 2

By default, all branch length entries are rescaled from 0 to 1 as proportion
to tree height (formatted to ease reading):

.. code-block:: shell

    #    C,   D,   A,   B,   E,   -,   -,   -,   -,   -  
      1.00,0.29,0.43,0.14,0.29,   0,   0,   0,   0,   0  # tip-to-node distance
      0.00,0.57,0.14,0.29,0.00,   0,   0,   0,   0,   0  # node-to-root distance
      0.43,0.29,0.29,0.14,0.29,   0,   0,   0,   0,   0  # tip edge length
      0.00,0.43,0.14,0.14,0.00,   0,   0,   0,   0,   0  # node edge length
         1,   1,   0,   1,   0,   0,   0,   0,   0,   0  # character 1
         0,   0,   1,   1,   1,   0,   0,   0,   0,   0  # character 2


CDV+S
^^^^^

CDV+S is used to encode phylogenetic-state information for trees of only
extant taxa. CDV+S has a similar structure to CBLV+S, except in two
principal ways. First, CDV+S uses total subclade diversity rather than
tip node with max distance-from-root-node to determine how to ladderize
the tree, which in turn determines which columns are associated with which
tip nodes. Second, because CDV+S is used for extant-only trees, it does not
need to report the redundant information about tip-to-node distances, as
the tip-to-root distances are equal among all tips (by definition). This
means that CDV+S does not contain a row with tip-to-node distances (the
first row of CBLV+S).


For example, the following Newick string for an ultrametric tree

.. code-block:: shell

    (((A:5,B:5):1,(C:3,D:3):3):1,E:7);

and associating the same character data as above with taxa A through E
yields the following CDV+S tensor:

.. code-block:: shell

    # C,D,A,B,E,-,-,-,-,-  
      0,4,1,2,0,0,0,0,0,0  # node-to-root distance
      3,2,2,1,2,0,0,0,0,0  # tip edge length
      0,3,1,1,0,0,0,0,0,0  # node edge length
      1,1,0,1,0,0,0,0,0,0  # character 1
      0,0,1,1,1,0,0,0,0,0  # character 2


Auxiliary data
^^^^^^^^^^^^^^

The auxiliary data tensor contains a panel of summary statistics extracted
from the inputted phylogeny and character data matrix for a given dataset.
Currently, phyddle generates the following summary statistics:

.. code-block:: shell

    tree_length       # sum of branch lengths
    num_taxa          # number of terminal taxa in tree/data
    root_age          # longest root-to-tip distance
    brlen_mean        # mean of branch lengths
    brlen_var         # variance of branch lengths
    brlen_skew        # skewness of branch lengths
    age_mean          # mean of internal node ages
    age_var           # variance of internal node ages
    age_skew          # skewness of internal node ages
    B1                # [`link <https://dendropy.org/library/treemeasure.html#dendropy.calculate.treemeasure.B1>`__]
    N_bar             # https://dendropy.org/library/treemeasure.html#dendropy.calculate.treemeasure.N_bar
    colless           # https://dendropy.org/library/treemeasure.html?highlight=colless#dendropy.calculate.treemeasure.colless_tree_imbalance
    treeness          # https://dendropy.org/library/treemeasure.html#dendropy.calculate.treemeasure.treeness
    f_dat_0           # frequency of taxa with character in state 0
    f_dat_1           # frequency of taxa with character in state 1
    ...



The auxiliary data tensor also contains any parameter values that shape the data-generating process, but can be treated as "known" rather than needing to be estimated. For example, the epidemiologists may assume they know the rate of infection recovery (gamma) based on public health or clinical data. Parameters may be treated as data by providing the labels for those parameters in the ``param_data`` entry of the config file. For example, setting ``'param_data' : [ 'recovery_0', 'S0_0' ]`` could be used to inform phyddle that the recovery rate and susceptible population sizes for location 0 are known for a phylogenetic SIR analysis. 


