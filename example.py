import trajectory, trajectory.aero, csv

import numpy as np
import pandas as pd

'''Rocket parameters'''
dry_mass = 60       # kg
length = 6.529      # m
radius = 98.5e-3    # m
ref_area = 0.0305128422 # m^2

"""Import data from CSV files"""

#Get approximate values for the rotational damping coefficients
c_damp_pitch = trajectory.aero.pitch_damping_coefficient(length, radius, fin_number = 21, area_per_fin = 0.07369928)
c_damp_roll = 0

#Import drag coefficients from RASAero II
aerodynamic_coefficients = trajectory.aero.RASAeroData("data/Martlet4RasAeroII.CSV", ref_area, c_damp_pitch, c_damp_roll)

#Import motor data - copied from Joe Hunt's simulation
with open('novus_sim_6/motor_out.csv') as csvfile:
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

'''Create the objects needed to initialise the Rocket object'''
mass_model = trajectory.CylindricalMassModel(dry_mass + np.array(prop_mass_data), motor_time_data, length, radius)
pulsar = trajectory.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data, gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)
launch_site = trajectory.LaunchSite(rail_length=10, rail_yaw=45, rail_pitch=45, alt=0, longi=0, lat=0, wind=[0,0,0])
parachute=trajectory.Parachute(13.9,0.78,1.13,0.78,1000,0)

"""Create the Rocket object"""
martlet4 = trajectory.Rocket(mass_model, pulsar, aerodynamic_coefficients, launch_site, h=0.05, variable=True,alt_poll_interval=1)

'''Run the simulation'''
#simulation_output = martlet4.run(max_time = 300, debug=True, to_json="output.json")

'''Example of how you can import data from a .csv file'''
imported_data = trajectory.from_json("output.json")

'''Plot the results'''
#trajectory.plot_launch_trajectory_3d(imported_data, martlet4, show_orientation=False) #Could have also used simulation_output instead of imported_data
#trajectory.plot_altitude_time(imported_data, martlet4)
trajectory.plot_ypr(imported_data, martlet4)
#trajectory.plot_aero(imported_data, martlet4)
trajectory.animate_orientation(imported_data)