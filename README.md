# CamPyRoS - A 6DOF Rocket Trajectory Simulator
[![DOI](https://zenodo.org/badge/308847422.svg)](https://zenodo.org/badge/latestdoi/308847422)

CamPyRoS (Cambridge Python Rocketry Simulator) is a Python package which provides fully featured rocket trajectory simulation including features like:
- 6 degrees of freedom (3 translational, 3 rotational)
- Monte Carlo stochastic analysis
- Aerodynamic heating model
- Use of live wind data
- Variable mass and moments of inertia models

For a fuller documatation see the docs [here](https://cuspaceflight.github.io/CamPyRoS-Docs/)

## Getting started
Download the repo and unzip into your preferred location, if you want to use this like a normal library (i.e. not just from within this directory you will need to move the CamPyRoS folder to your system path). Then install the dependencies as below:
### Anaconda (recommended)
Some dependencies are not properly configured to use pip so it is recommended to create a conda enviroment with the `enviroment.yml` file.

To do this:
`conda env create -f enviroment.yml -n <name>`
`conda ativate <name>`

### Pip (excludes 'ray' and 'iris' modules)
`pip install git+https://github.com/cuspaceflight/CamPyRoS.git`  

Alternative, you can download the repository onto your local disk, navigate to it, and run:
`pip install .`

## Usage

The repository contains some examples you can run:
- `example.ipynb` or `example.py` : Launch of a simple rocket (the Martlet 4).
- `Stats Model Example.ipynb` : Example of how to use the statistics model and stochastic analysis.
- `Aerodynamic Heating Example.ipynb` : Example of how to run an aerodynamic heating simulation.


## In progress
- **GUI:** An incomplete (and outdated) GUI has been made using Tkinter, and is in gui.py.
- **Slosh modelling:** Some slosh modelling functions have been put together in slosh.py, based on the following source - [The Dynamic Behavior of Liquids in Moving Containers, with Applications to Space Vehicle Technology](https://ntrs.nasa.gov/citations/19670006555).
- **Wind variability:** Statistical analysis of historic wind forecasts and obervations are analysed to create a Guassian difference profile to vary the wind in the statistical models (see wind-stats branch for a very poorly documented insight to current progress)


## Potential for expansion
- **Multistage rockets**
- **Fin cant, roll damping, and roll acceleration:** [OpenRocket Technical Documentation](http://openrocket.info/documentation.html)
- **CFD coupling:** [PyFoam](https://openfoamwiki.net/index.php/Contrib/PyFoam), [Simulations of 6-DOF Motion
with a Cartesian Method](https://pdfs.semanticscholar.org/ace3/5a61803390b0e0b70f6ca34492ad20a03e03.pdf)
- **Multiphysics coupling:** [PRECICE](https://www.precice.org/)

## Cite as
Daniel Gibbons, & Jago Strong-Wright. (2021, February 11). cuspaceflight/CamPyRoS: First release! (Version V1.0). Zenodo. http://doi.org/10.5281/zenodo.4535672

## Main References
[1] - [Stochastic Six-Degree-of-Freedom Flight Simulator for Passively Controlled High-Power Rockets](https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000051)
[2] - [STOCHASTIC FLIGHT SIMULATION APPLIED TO A SOUNDING ROCKET](https://sci-hub.do/10.2514/6.iac-04-a.1.07)
[3] - [Tangent ogive nose aerodynamic heating program: NQLD019](https://ntrs.nasa.gov/citations/19730063810)


## Additional References
[4] - [NASA Basic Considerations for Rocket Trajectory Simulation](https://apps.dtic.mil/sti/pdfs/AD0642855.pdf)
[5] - [SIX DEGREE OF FREEDOM DIGITAL SIMULATION MODEL FOR UNGUIDED FIN-STABILIZED ROCKETS](https://apps.dtic.mil/dtic/tr/fulltext/u2/452106.pdf)
[6] - [Trajectory Prediction for a Typical Fin Stabilized Artillery Rocket](https://journals.ekb.eg/article_23742_f19c1da1a61e78c1f5bb7ce58a7b30dd.pdf)
[7] - [Central Limit Theorem and Sample Size](https://www.umass.edu/remp/Papers/Smith&Wells_NERA06.pdf)
[8] - [Monte Carlo Simulations: Number of Iterations and Accuracy](https://apps.dtic.mil/dtic/tr/fulltext/u2/a621501.pdf)
[9] - [Method for Calculating Aerodynamic Heating on Sounding Rocket Tangent Ogive Noses](https://arc.aiaa.org/doi/abs/10.2514/3.62081)
[10] - [Six degree-of-freedom (6-DOF) Flight Simulation Check-cases](https://nescacademy.nasa.gov/flightsim/)




