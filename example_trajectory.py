import main,csv
from datetime import datetime
from main import StandardAtmosphere
import numpy as np
import matplotlib.pyplot as plt

'''Import data from CSV files'''

#Import drag coefficients from RasAero II
aerodynamic_coefficients = main.RasAeroData("Martlet4 RasAeroII.CSV")

#Import motor data - copied from Joe Hunt's simulation
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

dry_mass = 60
length = 6.529
radius = 98.5e-3

'''Create the objects needed to initialise the Rocket object'''
mass_model = main.CylindricalMassModel(dry_mass + np.array(prop_mass_data), motor_time_data, length, radius)
pulsar = main.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data, gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)
launch_site = main.LaunchSite(rail_length=5, rail_yaw=0, rail_pitch=0, alt=0, longi=0, lat=0, wind=[0,0,0], atmosphere=StandardAtmosphere)

'''Create the Rocket object'''
martlet4 = main.Rocket(mass_model, pulsar, aerodynamic_coefficients, launch_site, 0.05, False)

'''Run the simulation'''
simulation_output = main.run_simulation(martlet4)
#simulation_output.to_csv("results/results_%s.csv"%datetime.now().strftime("%m_%d_%Y_%H_%M_%S"), index=True, mode="w+")
print(min(simulation_output["h"]),max(simulation_output["h"]))

'''Plot the results'''
#main.plot_velocity(simulation_output)
#main.plot_altitude_time(simulation_output)
#main.plot_aero_forces(simulation_output)
#main.plot_orientation(simulation_output)
#main.plot_rot_acc(simulation_output)
#main.plot_position(simulation_output)

main.plot_orientation(simulation_output)
main.plot_inertial_trajectory_3d(simulation_output, show_orientation=True)
main.plot_launch_trajectory_3d(simulation_output, show_orientation=True)

main.animate_orientation(simulation_output)



