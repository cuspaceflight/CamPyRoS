"""Simple 3DOF Martlet 4 trajectory simulator"""

__copyright__ = """

    Copyright 2019 Joe Hunt

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

### Joe Hunt updated 20/06/19 ###
### All units SI unless otherwise stated ###

import csv
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Input parameters
###############################################################################

MASS_DRY = 60                     #rocket dry mass (kg)
DIAMETER = 0.197                  #rocket body DIAMETER (m)
LAUNCH_ALT = 615                    #launch altitude (msl)
ANGLE_RAIL = 89                   #launch rail angle, degrees
LENGTH_RAIL = 11                  #launch rail length (m)

vvel = 0                          #initial rocket vertical velocity (m s^-1)
hvel = 0                          #initial rocket horizontal velocity (m s^-1)
ground = 0                          #initial down-range distance (m)

###############################################################################
# Initialize simulation
###############################################################################

STEP = 0.01                        #time step (s)

# open motor performance file (output of motor_sim.py)
with open('motor_out.csv') as csvfile:
    motor_out = csv.reader(csvfile)

    (motor_time_data, prop_mass_data, cham_pres_data,
     throat_data, gamma_data, nozzle_efficiency_data,
     exit_pres_data, area_ratio_data) = [], [], [], [], [], [], [], []

    next(motor_out)
    for row in motor_out:
        motor_time_data.append(float(row[0]))
        prop_mass_data.append(float(row[1]))
        cham_pres_data.append(float(row[2]))
        throat_data.append(float(row[3]))
        gamma_data.append(float(row[4]))
        nozzle_efficiency_data.append(float(row[5]))
        exit_pres_data.append(float(row[6]))
        area_ratio_data.append(float(row[7]))

# open standard atmosphere data
with open('atmosphere_data.csv') as csvfile:
    standard_atmo_data = csv.reader(csvfile)
    adat, ddat, sdat, padat = [], [], [], []
    next(standard_atmo_data)
    for row in standard_atmo_data:
        adat.append(float(row[0]))
        ddat.append(float(row[1]))
        sdat.append(float(row[2]))
        padat.append(float(row[3]))

# import drag coeffcient as a function of Mach data
with open('drag_coefficient_data.csv') as csvfile:
    drag_coefficient_data = csv.reader(csvfile)
    machdat = []
    cddat = []
    next(drag_coefficient_data)
    for row in drag_coefficient_data:
        machdat.append(float(row[0]))
        cddat.append(float(row[1]))

# compute state of vehicle
apogee = False; GRAV_ACCEL = 9.81; alt = LAUNCH_ALT; stable = True; rail_left = False; time = 0
vel = np.sqrt((vvel**2)+(hvel**2))
pitch = np.radians(ANGLE_RAIL)
if alt < 80000:
    density = np.interp(alt, adat, ddat)
    vsound = np.interp(alt, adat, sdat)
    pres_static = np.interp(alt, adat, padat)
else:
    density, vsound = 0, float("inf")
mach = vel/vsound

# create empty lists to fill with output data
(time_data, alt_data, vel_data, acc_data, drag_data, thrust_data, mass_data,
 mach_data, ground_data, pitch_data) = [], [], [], [], [], [], [], [], [], []


###############################################################################
# Simulation loop
###############################################################################
while True:
    time += STEP

    # Update density(altitude), speed of sound(altitude), and cd(mach) from input data
    if alt < 80000:
        density = np.interp(alt, adat, ddat)
        vsound = np.interp(alt, adat, sdat)
        pres_static = np.interp(alt, adat, padat)

    else:
        density, vsound, pres_static = 0, float("inf"), 0

    cd = np.interp(mach, machdat, cddat)

    # Find current thrust
    if time < max(motor_time_data):
        pres_cham = np.interp(time, motor_time_data, cham_pres_data)
        dia_throat = np.interp(time, motor_time_data, throat_data)
        gamma = np.interp(time, motor_time_data, gamma_data)
        nozzle_efficiency = np.interp(time, motor_time_data, nozzle_efficiency_data)
        pres_exit = np.interp(time, motor_time_data, exit_pres_data)
        nozzle_area_ratio = np.interp(time, motor_time_data, area_ratio_data)

        # motor performance calculations
        area_throat = ((dia_throat/2)**2)*np.pi
        thrust = (area_throat*pres_cham*(((2*gamma**2/(gamma-1))
                                         *((2/(gamma+1))**((gamma+1)/(gamma-1)))
                                         *(1-(pres_exit/pres_cham)**((gamma-1)/gamma)))**0.5)
                 +(pres_exit-pres_static)*area_throat*nozzle_area_ratio)

        thrust *= nozzle_efficiency
    else:
        thrust = 0

    #update acceleration and integrate
    mass_prop = np.interp(time, motor_time_data, prop_mass_data)
    mass = mass_prop + MASS_DRY
    drag = 0.5*cd*density*(vel**2)*(((DIAMETER/2)**2)*np.pi)
    vacc = ((thrust*np.sin(pitch))/mass)-((drag*np.sin(pitch))/mass)-GRAV_ACCEL
    hacc = ((thrust*np.cos(pitch))/mass)-((drag*np.cos(pitch))/mass)
    vvel += vacc*STEP
    hvel += hacc*STEP
    vel = np.sqrt((vvel**2)+(hvel**2))
    acc = np.sqrt((vacc**2)+(hacc**2))

    if alt-LAUNCH_ALT < np.sin(pitch)*LENGTH_RAIL and apogee == False:
        pitch = np.radians(ANGLE_RAIL)
    else:
        pitch = np.arctan2(vvel, hvel)
        if rail_left == False:
            print('rail cleared at', vel, 'm/s', 'T/W:', thrust/(mass*GRAV_ACCEL))
            rail_left = True

    mach = vel/vsound
    alt += vvel*STEP
    ground += hvel*STEP

    # check for ground impact
    if alt < 0:
        break

    #update trajectory plot _data
    time_data.append(time)
    thrust_data.append(thrust)
    drag_data.append(-drag)
    alt_data.append(alt)
    vel_data.append(vel)
    acc_data.append(acc)
    mass_data.append(mass)
    mach_data.append(mach)
    ground_data.append(ground)
    pitch_data.append(np.degrees(pitch))


###############################################################################
# Print and plot results
###############################################################################

print('\nResults:\napogee:', (max(alt_data)-LAUNCH_ALT)/1000, 'km\nmax Mach:',
      max(mach_data))

print('Gross lift off mass:', mass_data[0], 'kg')

plt.figure(figsize=(9, 9))

plt.subplot(321)
plt.plot(ground_data, [a-LAUNCH_ALT for a in alt_data])
plt.xlabel('Downrange (m)')
plt.ylabel('Altitude (m)')
plt.ylim(0, max([a-LAUNCH_ALT for a in alt_data])*1.1)
plt.xlim(-1000, (max(ground_data)*1.1))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(322)
plt.plot(time_data, vel_data)
plt.xlabel('Time (s)')
plt.ylabel('Speed (ms-1)')
plt.ylim(min(vel_data)*1.3, max(vel_data)*1.3)
plt.axhline(y=0, color='k', linestyle='-')
plt.tight_layout()

plt.subplot(323)
plt.plot(time_data, pitch_data)
plt.xlabel('Time (s)')
plt.ylabel('Pitch (degrees)')
plt.ylim(min(pitch_data)*1.2, max(pitch_data)*1.2)
plt.axhline(y=0, color='k', linestyle='-')
plt.tight_layout()

plt.subplot(324)
plt.plot(time_data, thrust_data, 'r', label='thrust force')
plt.plot(time_data, drag_data, 'b', label='Drag force')
plt.plot(time_data, [-GRAV_ACCEL*m for m in mass_data], 'g', label='Weight force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.ylim(min(min(drag_data), min(thrust_data), min([-GRAV_ACCEL*m for m in mass_data]))*1.2,
         max(max(drag_data), max(thrust_data))*1.2)
plt.axhline(y=0, color='k', linestyle='-')
plt.legend()
plt.tight_layout()

plt.show()
