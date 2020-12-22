'''
Compared with page 4 of https://arc.aiaa.org/doi/pdf/10.2514/3.62081 - NASA Flight TND 889

At the peak heat transfer rate, reading off the NASA graph we have approximately:

alt = 6096 m
Vinf = 1280.16 m/s - this gives a Mach number of about 4.05 I think
q_turb = 44 Btu/ft^2/s = 499.69 kW/m^2
q_lam = 4 Btu/ft^2/s = 45.43 kW/m^2

All of these values are at a point between station 9 and 10
'''

import trajectory, trajectory.post, trajectory.aero, csv
import numpy as np
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, pos_i2alt
from martlet4 import martlet4

'''Set up the flight conditions'''
alt = 6096
Vinf = 1280.16
pos_i = pos_l2i([0, 0, alt], martlet4.launch_site, 0)
vel_i = vel_l2i([Vinf, 0, 0], martlet4.launch_site, 0)

trajectory_data = {"time" : [0, 1], 
                 "pos_i" : [pos_i, pos_i], 
                 "vel_i" : [vel_i, vel_i],
                 "b2imat": [None, None],
                 "w_b" : [None, None],
                 "events" : []}

'''Specify the nosecone and create the HeatTransfer analysis object'''
#The nosecone is the same as that used in Reference 8 (pg A-3) - xprime = 2.504 ft = 0.7632192 m , yprime = 0.25 ft = 0.0762 m
tangent_ogive = trajectory.post.TangentOgive(xprime = 0.7632192, yprime = 0.0762)
analysis = trajectory.post.HeatTransfer(tangent_ogive, trajectory_data, martlet4)

'''Run the simulation if you want'''
analysis.step(print_style="metric")

analysis.plot_fluid_properties(i=0)


