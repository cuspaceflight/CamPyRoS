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
statistical=main.StatisticalModel(mass_model,pulsar,aerodynamic_coefficients,launch_site,0.1,False,rail_pitch=1/30,rail_yaw=1/30,cop=0.05,ca=0.05,cn=0.05,thrust=0.03,gravity=0.01,mass=0.01,ixx=0.0,iyy=0.0,izz=0.0,thrust_alignment=np.array([1.0,0.0006,0.0006]),air_density=0.05,air_pressure=0.05,air_temp=0.05,wind=np.array([5.0,5.0,5.0]))
#statistical=main.StatisticalModel(mass_model,pulsar,aerodynamic_coefficients,launch_site,0.1,False)
statistical.run_model(100)


