.. _appendix:

Appendix
========

Glossary
--------

This section defines terms used by phyddle:

.. tabularcolumns:: p{0.1\linewidth}p{0.1\linewidth}p{0.1\linewidth}p{0.7\linewidth}
.. csv-table::
   :file: ./tables/glossary.csv
   :delim: |
   :header-rows: 1
   :widths: 20, 80
   :align: center
   :width: 100%
   :class: longtable


.. _setting_summary:

Table of Settings
-----------------

This table summarizes all settings currently available in phyddle.
The `Setting` column is the exact name of the string that appears in
the configuration file and command-line argument list. The `Step(s)` identifies
all steps that use the setting: [S]imulate, [F]ormat, [T]rain, [E]stimate, and
[P]lot. The `Type` column is the Python variable type expected for the setting.
The `Description` gives a brief description of what the setting does. Visit 
:ref:`Overview` to learn more about phyddle settings impact different pipeline
analysis steps. 

.. _table_phyddle_settings:

.. tabularcolumns:: p{0.1\linewidth}p{0.1\linewidth}p{0.1\linewidth}p{0.7\linewidth}
.. csv-table:: phyddle settings
   :file: ./tables/phyddle_settings.csv
   :header-rows: 1
   :widths: 10, 10, 10, 70
   :delim: |
   :align: center
   :width: 100%
   :class: longtable


.. _references:

References
----------

EE Goldberg, LT Lancaster, RH Ree. 2011. Phylogenetic inference of reciprocal
effects between geographic range evolution and diversification. Syst
Biol 60:451-465. doi: https://doi.org/10.1093/sysbio/syr046

S Lambert, J Voznica, H Morlon. 2022. Deep learning from phylogenies for
diversification analyses. bioRxiv. 2022.09.27.509667.
doi: https://doi.org/10.1101/2022.09.27.509667 

MJ Landis, A Thompson. 2024. phyddle: software for phylogenetic model
exploration with deep learning. bioRxiv 2024.08.06.606717.

Y Romano, E Patterson, E Candes. Conformalized quantile regression.
Adv NIPS, 32, 2019.
doi: https://doi.org/10.1101/2024.08.06.606717

A Thompson, B Liebeskind, EJ Scully, MJ Landis. 2023. Deep learning approaches
to viral phylogeography are fast and as robust as likelihood methods
to model misspecification. bioRxiv 2023.02.08.527714.
doi: https://doi.org/10.1101/2023.02.08.527714 

TG Vaughan, AJ Drummond. 2013. A stochastic simulator of birth–death
master equations with application to phylodynamics. Mol Biol Evol 30:1480–1493.
doi: https://doi.org/10.1093/molbev/mst057

J Voznica, A Zhukova, V Boskova, E Saulnier, F Lemoine, M Moslonka-Lefebvre,
O Gascuel. 2022. Deep learning from phylogenies to uncover the epidemiological
dynamics of outbreaks. Nat Commun 13:3896.
doi: https://doi.org/10.1038/s41467-022-31511-0


.. _about:

About
-----

Thanks for your interest in phyddle. The phyddle project emerged from a
phylogenetic deep learning study led by Ammon Thompson
(`paper <https://www.biorxiv.org/content/10.1101/2023.02.08.527714v2>`_).
The goal of phyddle is to provide its users with a generalizable pipeline
workflow for phylogenetic modeling and deep learning. This hopefully will make
it easier for phylogenetic model enthusiasts and developers to explore and
apply models that do not have tractable likelihood functions. It's also
intended for use by methods developers who want to characterize how deep
learning methods perform under different conditions for standard phylogenetic
estimation tasks.

The phyddle project is developed by `Michael Landis <https://landislab.org>`__
and `Ammon Thompson <https://scholar.google.com/citations?user=_EpmmTwAAAAJ&hl=en&oi=ao>`__.


.. _issues_feedback:

Issues & Feedback
-----------------

Please use `Issues <https://github.com/mlandis/phyddle/issues>`__ to report
bugs or request features that require modifying phyddle source code. Please
contact `Michael Landis <mailto:michael.landis@wustl.edu>`__ to request
troubleshooting support using phyddle.
