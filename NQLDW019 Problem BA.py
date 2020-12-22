'''
Used to check data against page 70 of the NASA "TANGENT OGIVE NOSE AERODYNAMIC HEATING PROGRAM - NQLDW019 (NASA)" documentation

Note that Problem BA in the NASA document is actually at an angle of attack of 10 degrees - which is ignored in my current Python script

'''

import trajectory, trajectory.aero, csv
import trajectory.post as post
import numpy as np
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, pos_i2alt

from martlet4 import martlet4

'''
Specify the nosecone, using:
xprime = 2.741 ft = 0.8354568 m
yprime = 0.5 ft = 0.1524 m 
'''
tangent_ogive = trajectory.post.TangentOgive(xprime = 0.8354568, yprime = 0.1524)
#tangent_ogive = post.TangentOgive(xprime = 2.741, yprime = 0.5)

#Freestream conditions:
ALT = 15240         #50000 ft
VINF = 1219.2       #4000 ft/s
ALPHA = 10 * np.pi/180

#Properties using standard atmosphere
TINF = 216.650              #K
PINF = 11597.3              #Pa
RHOINF = 0.186481           #kg/m3
speed_of_sound = 295.070    #m/s
MINF = VINF/speed_of_sound

#Set up the conditions for problem BA in the paper:
pos_i = pos_l2i([0, 0, ALT], martlet4.launch_site, 0)
vel_i = vel_l2i([VINF, 0, 0], martlet4.launch_site, 0)

trajectory_data = {"time" : [0, 1], 
                 "pos_i" : [pos_i, pos_i], 
                 "vel_i" : [vel_i, vel_i],
                 "b2imat": [None, None],
                 "w_b" : [None, None],
                 "events" : []}

analysis = post.HeatTransfer(tangent_ogive, trajectory_data, martlet4, starting_temperature = 300)
analysis.step(print_style = "FORTRAN")
