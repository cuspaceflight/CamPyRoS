import main,csv
from main import StandardAtmosphere
###############################################################################
#Import data from CSV files
###############################################################################

#Import drag coefficients - copied from Joe Hunt's simulation - will need more data for the other drag coefficients
#Maybe obtained from CFD, or OpenRocket/RASAero II if possible
with open('drag_coefficient_data.csv') as csvfile:
    drag_coefficient_data = csv.reader(csvfile)
    machdat = []
    cddat = []
    next(drag_coefficient_data)
    for row in drag_coefficient_data:
        machdat.append(float(row[0]))
        cddat.append(float(row[1]))


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


mass_data = {'dry_mass': 45.73, 'Izz':0.32, 'Ixx':86.8, 'Iyy':86.8}

    
###############################################################################
#Initialising and running the simulations
###############################################################################
    
pulsar = main.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data,
                          gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)

launch_site = main.LaunchSite(10, 1, 0 , 0, 0, 0)

martlet4 = main.Rocket(45.73, 86.8, 86.8, 0.32, pulsar, drag_coefficient_data, launch_site, 0.001, False)

martlet4.position_velocity()

#simulation_output = main.run_simulation(martlet4)

#main.plot_altitude_time(simulation_output)



