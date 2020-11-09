import main,csv
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


#mass_data = {'dry_mass': 45.73, 'Izz':0.32, 'Ixx':86.8, 'Iyy':86.8}

    
'''Create required objects, and run the simulation'''
    
pulsar = main.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data,
                          gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)

launch_site = main.LaunchSite(10, 1, 0 , 0, 0, 0)

martlet4 = main.Rocket(45.73, 86.8, 86.8, 0.32, pulsar, aerodynamic_coefficients, launch_site, 0.001, False)

#Create a time array
time = np.linspace(0, 50, 100)
thrust = np.zeros(len(time))
for i in range(len(time)):
    thrust[i] = np.linalg.norm(martlet4.thrust(time[i]))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)

ax.plot(time, thrust)
ax.grid()
ax.set_xlabel("Time / s")
ax.set_ylabel("Thrust / N")
plt.show()


