"""Simple 3DOF Martlet 4 trajectory simulator"""

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

h = 0.01                        #time step (s)
c=[0,1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0,1.0,1.0]
a=[[0,               0,              0,               0,            0,              0        ],
[1.0/5.0,         0,              0,               0,            0,              0        ],
[3.0/40.0,        9.0/40.0,       0,               0,            0,              0        ],
[44.0/45.0,      -56.0/15.0,      32.0/9.0,        0,            0,              0        ],
[19327.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,  0,              0        ],
[9017.0/3168.0,  -355.0/33.0,     46732.0/5247.0,  49.0/176.0,  -5103.0/18656.0, 0        ],
[35.0/384.0,      0,              500.0/1113.0,    125.0/192.0, -2187.0/6784.0,  11.0/84.0]]
b=[35.0/384.0,  0,  500.0/1113.0, 125.0/192.0, -2187.0/6784.0,  11.0/84.0,  0]
b_=[5179.0/57600.0,  0,  7571.0/16695.0,  393.0/640.0,  -92097.0/339200.0, 187.0/2100.0,  1.0/40.0]

atol=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #absolute error of each component of v and w
rtol=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #relative error of each component of v and w
sf=0.98 #Safety factor for h scaling

# open motor performance file (output of motor_sim.py)
with open('Motor/motor_out.csv') as csvfile:
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
def drag(vel):
    return 0.5*cd*density*(vel**2)*(((DIAMETER/2)**2)*np.pi)
while True:
    time += h
    #This is going to be discusting
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

    #This monstrosity took longer to program than the actual one but it would have taken longer to vectorise this
    k_1_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel)*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_1_h=h*(((thrust*np.cos(pitch))/mass)-((drag(vel)*np.cos(pitch))/mass))

    k_2_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel+a[1][0]*np.sqrt((k_1_v**2)+(k_1_h**2)))*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_2_h=h*((thrust*np.cos(pitch))/mass)-((drag(vel+a[1][0]*np.sqrt((k_1_v**2)+(k_1_h**2)))*np.cos(pitch))/mass)

    k_3_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel+a[2][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[2][1]*np.sqrt((k_2_v**2)+(k_2_h**2)))*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_3_h=h*((thrust*np.cos(pitch))/mass)-((drag(vel+a[2][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[2][1]*np.sqrt((k_2_v**2)+(k_2_h**2)))*np.cos(pitch))/mass)

    k_4_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel+a[3][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[3][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[3][2]*np.sqrt((k_3_v**2)+(k_3_h**2)))*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_4_h=h*((thrust*np.cos(pitch))/mass)-((drag(vel+a[3][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[3][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[3][2]*np.sqrt((k_3_v**2)+(k_3_h**2)))*np.cos(pitch))/mass)

    k_5_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel+a[4][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[4][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[4][2]*np.sqrt((k_3_v**2)+(k_3_h**2))+a[4][3]*np.sqrt((k_4_v**2)+(k_4_h**2)))*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_5_h=h*((thrust*np.cos(pitch))/mass)-((drag(vel+a[4][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[4][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[4][2]*np.sqrt((k_3_v**2)+(k_3_h**2))+a[4][3]*np.sqrt((k_4_v**2)+(k_4_h**2)))*np.cos(pitch))/mass)
    
    k_6_v=h*(((thrust*np.sin(pitch))/mass)-((drag(vel+a[5][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[5][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[5][2]*np.sqrt((k_3_v**2)+(k_3_h**2))+a[5][3]*np.sqrt((k_4_v**2)+(k_4_h**2))+a[5][4]*np.sqrt((k_5_v**2)+(k_5_h**2)))*np.sin(pitch))/mass)-GRAV_ACCEL)
    k_6_h=h*((thrust*np.cos(pitch))/mass)-((drag(vel+a[5][0]*np.sqrt((k_1_v**2)+(k_1_h**2))+a[5][1]*np.sqrt((k_2_v**2)+(k_2_h**2))+a[5][2]*np.sqrt((k_3_v**2)+(k_3_h**2))+a[5][3]*np.sqrt((k_4_v**2)+(k_4_h**2))+a[5][4]*np.sqrt((k_5_v**2)+(k_5_h**2)))*np.cos(pitch))/mass)

    vvel += b[0]*k_1_v+b[1]*k_2_v+b[2]*k_3_v+b[3]*k_4_v+b[4]*k_5_v+b[5]*k_6_v
    hvel += b[0]*k_1_h+b[1]*k_2_h+b[2]*k_3_h+b[3]*k_4_h+b[4]*k_5_h+b[5]*k_6_h

    vel = np.sqrt((vvel**2)+(hvel**2))
    acc = np.sqrt(((k_1_v/h)**2)+((k_1_h/h)**2))

    if alt-LAUNCH_ALT < np.sin(pitch)*LENGTH_RAIL and apogee == False:
        pitch = np.radians(ANGLE_RAIL)
    else:
        pitch = np.arctan2(vvel, hvel)
        if rail_left == False:
            print('rail cleared at', vel, 'm/s', 'T/W:', thrust/(mass*GRAV_ACCEL))
            rail_left = True

    mach = vel/vsound
    alt += vvel*h
    ground += hvel*h

    # check for ground impact
    if alt < 0:
        break

    #update trajectory plot _data
    time_data.append(time)
    thrust_data.append(thrust)
    drag_data.append(-drag(vel))
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
