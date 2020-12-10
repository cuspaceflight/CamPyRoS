import trajectory, csv

import numpy as np
import pandas as pd


"""Import data from CSV files"""
print("Importing CSV data")


#Import drag coefficients from RASAero II
aerodynamic_coefficients = trajectory.RasAeroData("data/Martlet4RasAeroII.CSV")

#Import motor data - copied from Joe Hunt's simulation
with open('novus_sim_6.1/motor_out.csv') as csvfile:
    motor_out = csv.reader(csvfile)

    (motor_time_data, prop_mass_data, cham_pres_data,
     throat_data, gamma_data, nozzle_efficiency_data,
     exit_pres_data, area_ratio_data, vmass_data, lden_data, lmass_data, fuel_mass_data) = [], [], [], [], [], [], [], [], [], [], [], []

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
        vmass_data.append(float(row[8]))
        lden_data.append(float(row[9]))
        lmass_data.append(float(row[10]))
        fuel_mass_data.append(float(row[11]))
        
        #This is a bit inefficient given that these are constants:
        DENSITY_FUEL = float(row[12])
        DIA_FUEL = float(row[13])
        LENGTH_PORT = float(row[14])

print("Finished importing CSV data")

'''Set up the mass model'''
dry_mass = 60                           # kg
rocket_length = 6.529                   # m
rocket_radius = 98.5e-3                 # m
rocket_wall_thickness = 1e-2            # m - This is just needed for the mass model
pos_tank_bottom = 4.456                 # m - Distance between the nose tip and the bottom of the nitrous tank
pos_solidfuel_bottom = 4.856+LENGTH_PORT


liquid_fuel = trajectory.LiquidFuel(lden_data, lmass_data, rocket_radius, pos_tank_bottom, motor_time_data)
solid_fuel = trajectory.SolidFuel(fuel_mass_data, DENSITY_FUEL, DIA_FUEL/2, LENGTH_PORT, pos_solidfuel_bottom, motor_time_data)
dry_mass_model = trajectory.HollowCylinder(rocket_radius, rocket_radius - rocket_wall_thickness, rocket_length, dry_mass)

mass_model = trajectory.HybridMassModel(rocket_length, solid_fuel, liquid_fuel, vmass_data, 
                                        dry_mass_model.mass, dry_mass_model.ixx, dry_mass_model.iyy, dry_mass_model.izz, 
                                        dry_cog = rocket_length/2)

mass_model = trajectory.CylindricalMassModel(dry_mass + np.array(prop_mass_data), motor_time_data, rocket_length, rocket_radius)

'''Create the other objects needed to initialise the Rocket object'''

pulsar = trajectory.Motor(motor_time_data, 
                          prop_mass_data, 
                          cham_pres_data, 
                          throat_data, 
                          gamma_data, 
                          nozzle_efficiency_data, 
                          exit_pres_data, area_ratio_data)

launch_site = trajectory.LaunchSite(rail_length=10, 
                                    rail_yaw=0, 
                                    rail_pitch=0, 
                                    alt=0, 
                                    longi=0, 
                                    lat=0, 
                                    wind=[4.94975,4.94975,0])

parachute=trajectory.Parachute(main_s = 13.9,
                               main_c_d = 0.78,
                               drogue_s = 1.13,
                               drogue_c_d = 0.78,
                               main_alt = 1000,
                               attatch_distance = 0)

"""Create the Rocket object"""
martlet4 = trajectory.Rocket(mass_model, pulsar, aerodynamic_coefficients, launch_site, h=0.05, variable=True,alt_poll_interval=1,parachute=parachute)

'''Run the simulation'''
simulation_output = martlet4.run(max_time = 400, debug=True, to_json="output.json")

'''Example of how you can import data from a .csv file'''
imported_data = trajectory.from_json("output.json")

'''Plot the results'''
#trajectory.plot_launch_trajectory_3d(imported_data, martlet4, show_orientation=True) #Could have also used simulation_output instead of imported_data
#trajectory.plot_altitude_time(imported_data, martlet4)
#trajectory.plot_ypr(imported_data, martlet4)

'''Extra plots you could make'''
trajectory.plot_mass(imported_data, martlet4)