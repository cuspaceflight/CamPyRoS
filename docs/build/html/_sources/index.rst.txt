CUSF 6DOF Rocket Trajectory Simulation documentation
====================================================
*Insert name* is a python package which provides fully featured rocket trajectory simulation including features like:

- 6 degrees of freedom (3 translational, 3 rotational)
- Monte carlo stochastic analysis
- Aerodynamic heating model
- Variable mass and moments of inertia models  

Getting Started
^^^^^^^^^^^^^^^
Dependencies require conda installation since not all are available on pip. An enviroment yaml is provided so install these with::
    conda env create -f enviroment.yml -n <name>
And then activate::
    conda activate <name>

The examples will then help you get going:
The repository contains some examples you can run:  

- `example.ipynb <https://github.com/CUSF-Simulation/6DOF-Trajectory-Simulation/example.ipynb>`_ or `example.py` : Launch of a simple rocket (the Martlet 4).  
- `Stats Model Example.ipynb <https://github.com/CUSF-Simulation/6DOF-Trajectory-Simulation/blob/main/Stats%20Model%20Analysis%20Example.ipynb>`_ : Example of how to use the statistics model and stochastic analysis.  
- `Aerodynamic Heating Example.ipynb <https://github.com/CUSF-Simulation/6DOF-Trajectory-Simulation/blob/main/Aerodynamic%20Heating%20Example.ipynb>`_ : Example of how to run an aerodynamic heating simulation.  


Technical Documentation
^^^^^^^^^^^^^^^^^^^^^^^
Full technical documentation is coming soon


Content
=======
.. toctree::
   :maxdepth: 3
   :caption: Contents

   help.rst
   trajectory.rst
   develop.rst
   licences.rst



Index and moduals
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
