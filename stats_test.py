import trajectory,csv
import numpy as np
import trajectory.statistical as stats 

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

aero_file="data/Martlet4RasAeroII.CSV"
aero_error = {"COP":0.05,"CN":0.05,"CA":0.05}
mass_model_vars={"dry_mass":[60,0.01],"prop_mass":[np.array(prop_mass_data),0.01],"time_data":[motor_time_data,0], "length":[6.529,0.015], "radius":[98.5e-3,0.015]}
launch_site_vars={"rail_length":[10,.1], "rail_yaw":[0,0.03], "rail_pitch":[0,0.03], "alt":[0,1], "longi":[0,0.01], "lat":[0,0.01]}
motor_base = trajectory.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data, gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)
thrust_error = 0.03
thrust_alignment_error = 0.0006 #~2'
parachute_vars={"main_s":[13.9,0.05],"main_c_d":[0.78,0.05],"drogue_s":[1.13,0.05],"drogue_c_d":[0.78,0.05],"main_alt":[1000,0.05],"attatch_distance":[0,0]}
env_vars={"gravity":0.01,"pressure":0.05,"density":0.05,"speed_of_sound":0.05}

"""aero_file="data/Martlet4RasAeroII.CSV"
aero_error = {"COP":0.0,"CN":0.0,"CA":0.0}
mass_model_vars={"dry_mass":[60,0.0],"prop_mass":[np.array(prop_mass_data),0.0],"time_data":[motor_time_data,0], "length":[6.529,0.0], "radius":[98.5e-3,0.0]}
launch_site_vars={"rail_length":[10,0.0], "rail_yaw":[0,0.0], "rail_pitch":[0,0.0], "alt":[1,0], "longi":[0.1160127,0.0], "lat":[52.2079404,0.0]}
motor_base = trajectory.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data, gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)
thrust_error = 0.0
thrust_alignment_error = 0.000 #~2'
parachute_vars={"main_s":[13.9,0.0],"main_c_d":[0.78,0.0],"drogue_s":[1.13,0.0],"drogue_c_d":[0.78,0.0],"main_alt":[1000,0.0],"attatch_distance":[0,0]}
env_vars={"gravity":0.0,"pressure":0.0,"density":0.0,"speed_of_sound":0.0}"""

model = stats.StatisticalModel(launch_site_vars,mass_model_vars,aero_file,aero_error,motor_base,thrust_error,thrust_alignment_error,parachute_vars,env_vars)

model.run_model(100)