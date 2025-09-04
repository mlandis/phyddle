.. _Examples:

Examples
========

This page contains examples of phyddle projects for different phylogenetic
modeling scenarios. Each project includes a config file and a simulation script.
The config files specify how the pipeline is run (e.g. number of simulations
to run, number of training epochs, which parameters to estimate, etc.). The
simulation scripts define the exact simulation settings (e.g. numbers of taxa,
numbers of taxa, rate constraints, etc.).

We chose a variety of models and languages to help new users get started.
We encourage you to modify the scripts to suit your needs. However, be
sure to carefully read the :ref:`Overview` and :ref:`Safe_Usage` pages
before to understand the implications of your changes and how to improve
analysis performance.


.. _example_bisse_r:

BiSSE with R
------------

*Description:* Simulates trees and character data under binary state-dependent
speciation and extinction (BiSSE) models. Allows speciation rates to be
state-independent (:math:`\lambda_1 = \lambda_2`) or state-independent
(:math:`\lambda_1 \neq \lambda_2`). Extinction rates and transition
rates assumed to be state-independent for simplicity
(:math:`\mu_1 = \mu_2` and :math:`q_{12} = q_{21}`). Estimates model rates
and model type.

| *Config:* `config.py <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/bisse_r/config.py>`__
| *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/bisse_r/sim_bisse.R>`__ 
| *Dependencies:* `R <https://www.r-project.org/>`__, `ape <https://cran.r-project.org/web/packages/ape/>`__, `castor <https://cran.r-project.org/web/packages/castor/index.html>`__




.. _example_levy_r:

Continuous trait models with R
-------------------------------------

*Description:* Simulates trees under a birth-death model. Simulates
continuous-valued trait data under Brownian motion, Ornstein-Uhlenbeck,
Early Burst, and Lévy process models. Estimates model parameters, ancestral
trait value (at root), and model type.

| *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/levy_r/config.py>`__
| *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/levy_r/sim_levy.R>`__ 
| *Dependencies:* `R <https://www.r-project.org/>`__, `ape <https://cran.r-project.org/web/packages/ape/>`__, `castor <https://cran.r-project.org/web/packages/castor/index.html>`__, `statmod <https://cran.r-project.org/web/packages/statmod/index.html>`__, `pulsR <https://github.com/Schraiber/pulsR>`__



.. _example_sirm_master:

SIR + Migration with MASTER
---------------------------

*Description:* Simulates trees and character data under an SIR compartment
model with migration among locations. Can be configured to simulate
only during the exponential growth phase or extended beyond that phase.
Estimates basic reproduction number, migration rate, sampling rate, and
ancestral location state. 

| *Exponential phase*
|   *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/sirm_exp_phase_master/config.py>`__
|   *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/sirm_exp_phase_master/sim_sirm_exp.py>`__ 
| *All phases*
|   *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/sirm_all_phases_master/config.py>`__
|   *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/sirm_all_phases_master/sim_sirm_all_phases.py>`__ 
| *Dependencies:* `BEAST2 <http://www.beast2.org/>`__, `MASTER <https://tgvaughan.github.io/MASTER/>`__, `masterpy <https://pypi.org/project/masterpy/>`__


.. _example_bisse_timehet_pj:

Time-heterogeneous BiSSE with PhyloJunction
-------------------------------------------

*Description:* Simulates trees and character data under binary state-dependent
speciation and extinction (BiSSE) models. Allows speciation rates to be
state-independent (:math:`\lambda_1 = \lambda_2`) or state-independent
(:math:`\lambda_1 \neq \lambda_2`). Extinction rates and transition
rates assumed to be state-independent for simplicity
(:math:`\mu_1 = \mu_2` and :math:`q_{12} = q_{21}`). All species experience
shift into a new rates at a random time. Estimates model rates
and time of rate shift event.

| *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/bisse_timehet_pj/config.py>`__
| *Simulator script (front-end):* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/bisse_timehet_pj/sim_bisse_timehet.py>`__ 
| *Simulator script (back-end):* `bisse_timehet.pj <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/bisse_timehet_pj/bisse_timehet.pj>`__
| *Dependencies:* `PhyloJunction <https://pypi.org/project/phylojunction/>`__


.. _example_geosse_revbayes:

GeoSSE with RevBayes
--------------------

*Description:* Simulates trees and character data a geographic
state-dependent speciation and extinction (GeoSSE) model. Assumes all events
of the same type have equal rates (e.g. :math:`d_{ij} = d_{kl}`).
Estimates model parameters.

| *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/geosse_revbayes/config.py>`__
| *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/geosse_revbayes/sim_geosse.Rev>`__ 
| *Dependencies:* `RevBayes <https://revbayes.github.io/>`__

.. _example_mol_revbayes:


K80 substitution model with RevBayes
------------------------------------

*Description:* Simulates trees under a birth-death model. Simulates
molecular sequences under a Kimura (1980) substitution model with
transition-transversion biases. Estimates model parameters.

| *Config:* `config.py <https://github.com/mlandis/phyddle/tree/main/workspace/mol_revbayes/config.py>`__
| *Simulator script:* `sim_bisse.R <https://raw.githubusercontent.com/mlandis/phyddle/refs/heads/main/workspace/mol_revbayes/sim_mol.Rev>`__ 
| *Dependencies:* `RevBayes <https://revbayes.github.io/>`__
