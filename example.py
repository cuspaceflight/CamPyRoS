import trajectory, csv,time

import numpy as np
import pandas as pd

'''Import motor data to use for the mass model - copied from Joe Hunt's simulation'''
with open('novus_sim_6.1/motor_out.csv') as csvfile:
    motor_out = csv.reader(csvfile)

    (motor_time_data, prop_mass_data, cham_pres_data,
     throat_data, gamma_data, nozzle_efficiency_data,
     exit_pres_data, area_ratio_data, vden_data, vmass_data, lden_data, lmass_data, fuel_mass_data) = [], [], [], [], [], [], [], [], [], [], [], [], []

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
        vden_data.append(float(row[8]))
        vmass_data.append(float(row[9]))
        lden_data.append(float(row[10]))
        lmass_data.append(float(row[11]))
        fuel_mass_data.append(float(row[12]))
        
        #This is a bit inefficient given that these are constants, (we only need to record them once):
        DENSITY_FUEL = float(row[13])
        DIA_FUEL = float(row[14])
        LENGTH_PORT = float(row[15])

'''Rocket parameters'''
DRY_MASS = 60                               # kg
ROCKET_LENGTH = 6.529                       # m
ROCKET_RADIUS = 98.5e-3                     # m
ROCKET_WALL_THICKNESS = 1e-2                # m - This is just needed for the mass model
POS_TANK_BOTTOM = 4.456                     # m - Distance between the nose tip and the bottom of the nitrous tank
POS_SOLIDFUEL_BOTTOM = 4.856+LENGTH_PORT    # m - Distance between the nose tip and bottom of the solid fuel grain 
REF_AREA = 0.0305128422                     # m^2 - Reference area for aerodynamic coefficients

'''Set up aerodynamic properties'''
#Get approximate values for the rotational damping coefficients
C_DAMP_PITCH = trajectory.pitch_damping_coefficient(ROCKET_LENGTH, ROCKET_RADIUS, fin_number = 4, area_per_fin = 0.07369928)
C_DAMP_ROLL = 0

#Import drag coefficients from RASAero II
aerodynamic_coefficients = trajectory.AeroData.from_rasaero("data/Martlet4RasAeroII.CSV", REF_AREA, C_DAMP_PITCH, C_DAMP_ROLL)

'''Set up the mass model'''
liquid_fuel = trajectory.LiquidFuel(lden_data, lmass_data, ROCKET_RADIUS, POS_TANK_BOTTOM, motor_time_data)
solid_fuel = trajectory.SolidFuel(fuel_mass_data, DENSITY_FUEL, DIA_FUEL/2, LENGTH_PORT, POS_SOLIDFUEL_BOTTOM, motor_time_data)
dry_mass_model = trajectory.HollowCylinder(ROCKET_RADIUS, ROCKET_RADIUS - ROCKET_WALL_THICKNESS, ROCKET_LENGTH, DRY_MASS)

mass_model = trajectory.HybridMassModel(ROCKET_LENGTH, solid_fuel, liquid_fuel, vmass_data, 
                                        dry_mass_model.mass, dry_mass_model.ixx(), dry_mass_model.iyy(), dry_mass_model.izz(), 
                                        dry_cog = ROCKET_LENGTH/2)

'''Create the other objects needed to initialise the Rocket object'''
pulsar = trajectory.Motor.from_novus('novus_sim_6.1/motor_out.csv')
'''
launch_site = trajectory.LaunchSite(rail_length=10, 
                                    rail_yaw=0, 
                                    rail_pitch=0, 
                                    alt=1, 
                                    longi=0.1160127, 
                                    lat=52.2079404, 
                                    variable_wind=True,
                                    forcast_plus_time="016",
                                    run_date="20201216",
                                    fast_wind=False)
'''
launch_site = trajectory.LaunchSite(rail_length=5, 
                                    rail_yaw=0, 
                                    rail_pitch=0, 
                                    alt=1, 
                                    longi=0, 
                                    lat=0, 
                                    variable_wind=False,
                                    default_wind=np.array([5,0,0]))#Use this version if you don't want to use the real wind (e.g. to test something else)

parachute = trajectory.Parachute(main_s = 13.9,
                                 main_c_d = 0.78,
                                 drogue_s = 1.13,
                                 drogue_c_d = 0.78,
                                 main_alt = 1000,
                                 attach_distance = 0)

"""Create the Rocket object"""
martlet4 = trajectory.Rocket(mass_model, pulsar, aerodynamic_coefficients, launch_site, h=0.05, variable=True, alt_poll_interval=1, parachute=parachute)

'''Run the simulation'''
t=time.time()
simulation_output = martlet4.run(debug=True,to_json="output.json")
print(time.time()-t)

'''Example of how you can import data from a .csv file'''
imported_data = trajectory.from_json("output.json")

'''Plot the results'''

trajectory.plot_launch_trajectory_3d(imported_data, martlet4, show_orientation=False) #Could have also used simulation_output instead of imported_data
trajectory.plot_altitude_time(imported_data, martlet4)
trajectory.plot_ypr(imported_data, martlet4)
#trajectory.animate_orientation(imported_data)

'''Extra plots you could make'''
#trajectory.plot_mass(imported_data, martlet4)
trajectory.plot_aero(imported_data, martlet4)
