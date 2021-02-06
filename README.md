# 6 DOF Rocket Simulation - Cambridge University Spaceflight
## Description
It's a six degree of freedom rocket simulator! Currently includes:
- 6 degrees of freedom (3 translational, 3 rotational)
- Stochastic analysis
- Aerodynamic heating model
- Variable mass and moments of inertia models  

## Getting started
The requirements.txt contains a list of all dependencies, however using `pip install -r requirements.txt` can be problematic due to some of the modules used. It is therefore recommended that you install dependencies by adding `environment.yml` as an environment in Anaconda.  

**insert info here on how to use environment.yml**


## Usage
See the docs (**insert link to the docs here**)  

The repository contains some examples you can run:  
- `example.py` : Launch of a simple rocket (the Martlet 4).  
- `Stats Model Analysis Example.ipynb` : Example of how to use the statistics model and stochastic analysis.  
- `Aerodynamic Heating Example.ipynb` : Example of how to run an aerodynamic heating simulation.  


## Potential for expansion
- **Multistage rockets**
- **GUI:** An incomplete (and outdated) GUI has been made using Tkinter, and is in gui.py.
- **Fin cant, roll damping, and roll acceleration:** [OpenRocket Technical Documentation](http://openrocket.info/documentation.html)
- **Slosh modelling:** Some slosh modelling functions have been put together in slosh.py, based on the following source - [The Dynamic Behavior of Liquids in Moving Containers, with Applications to Space Vehicle Technology](https://ntrs.nasa.gov/citations/19670006555).
- **CFD coupling:** [PyFoam](https://openfoamwiki.net/index.php/Contrib/PyFoam), [Simulations of 6-DOF Motion
with a Cartesian Method](https://pdfs.semanticscholar.org/ace3/5a61803390b0e0b70f6ca34492ad20a03e03.pdf)
- **Multiphysics coupling:** [PRECICE](https://www.precice.org/)
- **Comparison with check cases:** [Six degree-of-freedom (6-DOF) Flight Simulation Check-cases](https://nescacademy.nasa.gov/flightsim/)  


## License
**we need to decide on a license**

## Main References:
[1] - [Stochastic Six-Degree-of-Freedom Flight Simulator for Passively Controlled High-Power Rockets](https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000051)  
[2] - [STOCHASTIC FLIGHT SIMULATION APPLIED TO A SOUNDING ROCKET](https://sci-hub.do/10.2514/6.iac-04-a.1.07)  
[3] - [Tangent ogive nose aerodynamic heating program: NQLD019](https://ntrs.nasa.gov/citations/19730063810)  

## Additional References:
[4] - [NASA Basic Considerations for Rocket Trajectory Simulation](https://apps.dtic.mil/sti/pdfs/AD0642855.pdf)  
[5] - [SIX DEGREE OF FREEDOM DIGITAL SIMULATION MODEL FOR UNGUIDED FIN-STABILIZED ROCKETS](https://apps.dtic.mil/dtic/tr/fulltext/u2/452106.pdf)  
[6] - [Trajectory Prediction for a Typical Fin Stabilized Artillery Rocket](https://journals.ekb.eg/article_23742_f19c1da1a61e78c1f5bb7ce58a7b30dd.pdf)  
[7] - [Central Limit Theorem and Sample Size](https://www.umass.edu/remp/Papers/Smith&Wells_NERA06.pdf)  
[8] - [Monte Carlo Simulations: Number of Iterations and Accuracy](https://apps.dtic.mil/dtic/tr/fulltext/u2/a621501.pdf)  
[9] - [Method for Calculating Aerodynamic Heating on Sounding Rocket Tangent Ogive Noses](https://arc.aiaa.org/doi/abs/10.2514/3.62081)  




