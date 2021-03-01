CamPyRoS - A 6DOF Rocket Trajectory Simulator
=============================================
.. image:: https://zenodo.org/badge/308847422.svg
   :target: https://zenodo.org/badge/latestdoi/308847422

CamPyRoS (Cambridge Python Rocketry Simulator) is a Python package which provides fully featured rocket trajectory simulation including features like:

- 6 degrees of freedom (3 translational, 3 rotational)
- Monte Carlo stochastic analysis
- Aerodynamic heating model
- Use of live wind data
- Variable mass and moments of inertia models

Getting started
===============
Currently not all dependancies are supported by the same install methods so the easiest install doesn't contain the full functionality. To install the core library:

.. code-block:: python
  :linenos:
  
  pip install git+https://github.com/cuspaceflight/CamPyRoS.git 

The "wind" and "statistics" modules will not run. Statistics has a dependancy not fully supported by windows, to install it:

`pip install ray` on most platforms, for Windows problems see `here <https://docs.ray.io/en/master/installation.html>`_.

Wind depends on a library called iris which can only be installed with conda:

.. code-block:: python
  :linenos:
  
  conda install iris, iris_grib

.. note::
    Note: after some more testing this won't work for anyone on Windows, I am in the process fo sorting this out (the problem is a codec dependancy which there is no way for us to fix) by writing an library to read the other GFS distrobution method, see `getgfs <https://github.com/jagoosw/getgfs>`_. You can still run and contribute without using the wind module.

This may then demand you install another library when you try to run it:

.. code-block:: python
  :linenos:
  
  pip install eccodes-python

Alternativly you can download this repository and move it to either your system path or somewhere you will exclusivly use it from then:

.. code-block:: python
  :linenos:

    conda env create -f enviroment.yml -n <name>
    conda activate <name>

From within the downloaded folder.

Usage
=====

The repository contains some examples you can run:
- `example.ipynb` or `example.py` : Launch of a simple rocket (the Martlet 4).
- `Stats Model Example.ipynb` : Example of how to use the statistics model and stochastic analysis.
- `Aerodynamic Heating Example.ipynb` : Example of how to run an aerodynamic heating simulation.

Helping out
===========
If you would like to contribute please have a look at the `guidelines <https://github.com/cuspaceflight/CamPyRoS/blob/main/CONTRIBUTING.md>`_


In progress
===========
- **GUI:** An incomplete (and outdated) GUI has been made using Tkinter, and is in gui.py.
- **Slosh modelling:** Some slosh modelling functions have been put together in slosh.py, based on the following source - `The Dynamic Behavior of Liquids in Moving Containers, with Applications to Space Vehicle Technology <https://ntrs.nasa.gov/citations/19670006555>`_.
- **Wind variability:** Statistical analysis of historic wind forecasts and obervations are analysed to create a Guassian difference profile to vary the wind in the statistical models (see wind-stats branch for a very poorly documented insight to current progress)


Potential for expansion
=======================
- **Multistage rockets**
- **Fin cant, roll damping, and roll acceleration:** `OpenRocket Technical Documentation <http://openrocket.info/documentation.html>`_
- **CFD coupling:** `PyFoam <https://openfoamwiki.net/index.php/Contrib/PyFoam>`_, `Simulations of 6-DOF Motion with a Cartesian Method <https://pdfs.semanticscholar.org/ace3/5a61803390b0e0b70f6ca34492ad20a03e03.pdf>`_
- **Multiphysics coupling:** `PRECICE <https://www.precice.org/>`_

Cite as
=======
Daniel Gibbons, & Jago Strong-Wright. (2021, February 11). cuspaceflight/CamPyRoS: First release! (Version V1.0). Zenodo. http://doi.org/10.5281/zenodo.4535672

References
==========

[1] - `Stochastic Six-Degree-of-Freedom Flight Simulator for Passively Controlled High-Power Rockets <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000051>`_

[2] - `Tangent ogive nose aerodynamic heating program: NQLD019 <https://ntrs.nasa.gov/citations/19730063810>`_

[3] - `NASA Basic Considerations for Rocket Trajectory Simulation <https://apps.dtic.mil/sti/pdfs/AD0642855.pdf>`_

[4] - `SIX DEGREE OF FREEDOM DIGITAL SIMULATION MODEL FOR UNGUIDED FIN-STABILIZED ROCKETS <https://apps.dtic.mil/dtic/tr/fulltext/u2/452106.pdf>`_

[5] - `Trajectory Prediction for a Typical Fin Stabilized Artillery Rocket <https://journals.ekb.eg/article_23742_f19c1da1a61e78c1f5bb7ce58a7b30dd.pdf>`_

[6] - `Central Limit Theorem and Sample Size <https://www.umass.edu/remp/Papers/Smith&Wells_NERA06.pdf>`_

[7] - `Monte Carlo Simulations: Number of Iterations and Accuracy <https://apps.dtic.mil/dtic/tr/fulltext/u2/a621501.pdf>`_

[8] - `Method for Calculating Aerodynamic Heating on Sounding Rocket Tangent Ogive Noses <https://arc.aiaa.org/doi/abs/10.2514/3.62081>`_

[9] - `Six degree-of-freedom (6-DOF) Flight Simulation Check-cases <https://nescacademy.nasa.gov/flightsim/>`_



Technical Documentation
=======================
Full technical documentation is coming soon


Content
=======
.. toctree::
   :maxdepth: 3
   :caption: Contents

   help.rst
   campyros.rst
   develop.rst
   licences.rst



Index and moduals
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
