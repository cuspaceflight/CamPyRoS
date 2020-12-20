import trajectory, trajectory.post, trajectory.aero, csv
import numpy as np

'''Import motor data - copied from Joe Hunt's simulation'''
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
        
        #This is a bit inefficient given that these are constants, (we only need to record them once):
        DENSITY_FUEL = float(row[12])
        DIA_FUEL = float(row[13])
        LENGTH_PORT = float(row[14])

'''Rocket parameters'''
dry_mass = 60                               # kg
rocket_length = 6.529                       # m
rocket_radius = 98.5e-3                     # m
rocket_wall_thickness = 1e-2                # m - This is just needed for the mass model
pos_tank_bottom = 4.456                     # m - Distance between the nose tip and the bottom of the nitrous tank
pos_solidfuel_bottom = 4.856+LENGTH_PORT    # m - Distance between the nose tip and bottom of the solid fuel grain 
ref_area = 0.0305128422                     # m^2 - Reference area for aerodynamic coefficients

'''Set up aerodynamic properties'''
#Get approximate values for the rotational damping coefficients
c_damp_pitch = trajectory.aero.pitch_damping_coefficient(rocket_length, rocket_radius, fin_number = 4, area_per_fin = 0.07369928)
c_damp_roll = 0

#Import drag coefficients from RASAero II
aerodynamic_coefficients = trajectory.aero.RASAeroData("data/Martlet4RasAeroII.CSV", ref_area, c_damp_pitch, c_damp_roll)

'''Set up the mass model'''
liquid_fuel = trajectory.LiquidFuel(lden_data, lmass_data, rocket_radius, pos_tank_bottom, motor_time_data)
solid_fuel = trajectory.SolidFuel(fuel_mass_data, DENSITY_FUEL, DIA_FUEL/2, LENGTH_PORT, pos_solidfuel_bottom, motor_time_data)
dry_mass_model = trajectory.HollowCylinder(rocket_radius, rocket_radius - rocket_wall_thickness, rocket_length, dry_mass)

mass_model = trajectory.HybridMassModel(rocket_length, solid_fuel, liquid_fuel, vmass_data, 
                                        dry_mass_model.mass, dry_mass_model.ixx(), dry_mass_model.iyy(), dry_mass_model.izz(), 
                                        dry_cog = rocket_length/2)

'''Create the other objects needed to initialise the Rocket object'''
pulsar = trajectory.Motor(motor_time_data, 
                          prop_mass_data, 
                          cham_pres_data, 
                          throat_data, 
                          gamma_data, 
                          nozzle_efficiency_data, 
                          exit_pres_data, 
                          area_ratio_data)

launch_site = trajectory.LaunchSite(rail_length=10, 
                                    rail_yaw=0, 
                                    rail_pitch=0, 
                                    alt=0, 
                                    longi=0, 
                                    lat=0, 
                                    wind=[4.94975,4.94975,0])

parachute = trajectory.Parachute(main_s = 13.9,
                                 main_c_d = 0.78,
                                 drogue_s = 1.13,
                                 drogue_c_d = 0.78,
                                 main_alt = 1000,
                                 attatch_distance = 0)

"""Create the Rocket object"""
martlet4 = trajectory.Rocket(mass_model, pulsar, aerodynamic_coefficients, launch_site, h=0.05, variable=True, alt_poll_interval=1, parachute=parachute)

'''Import the trajectory data'''
imported_data = trajectory.from_json("output.json")

'''Specify the nosecone and create the HeatTransfer analysis object'''
#The nosecone is the same as that used in Reference 8 (pg A-3) - xprime = 2.504 ft, yprime = 0.25 ft
tangent_ogive = trajectory.post.TangentOgive(xprime = 0.7632192, yprime = 0.0762)
analysis = trajectory.post.HeatTransfer(tangent_ogive, imported_data, martlet4)

'''Run the simulation if you want'''
analysis.run(iterations = 300)

'''Import the aerodynamic heating data and plot it'''
analysis.from_json("aero_heating_output.json")
analysis.plot_heat_transfer_rates(imax=300)

'''
At the peak heat transfer rate, reading off the NASA graph we have approximately:

alt = 6096 m
Vinf = 1280.16 m/s - this gives a Mach number of about 4.05 I think
q_turb = 44 Btu/ft^2/s = 499.69 kW/m^2
q_lam = 4 Btu/ft^2/s = 45.43 kW/m^2

All of these values are at a point between station 9 and 10

My code outputs:
i=0 station=1 q_lam=nan kW/m^2 q_turb=inf kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=2 q_lam=2171.25 kW/m^2 q_turb=1579.68 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=3 q_lam=1630.29 kW/m^2 q_turb=1301.69 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=4 q_lam=1263.88 kW/m^2 q_turb=1133.50 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=5 q_lam=1016.33 kW/m^2 q_turb=1008.18 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=6 q_lam=835.74 kW/m^2 q_turb=906.18 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=7 q_lam=696.25 kW/m^2 q_turb=819.18 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=8 q_lam=584.11 kW/m^2 q_turb=742.86 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=9 q_lam=491.34 kW/m^2 q_turb=674.72 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=10 q_lam=412.97 kW/m^2 q_turb=613.17 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=11 q_lam=461.07 kW/m^2 q_turb=674.50 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=12 q_lam=460.18 kW/m^2 q_turb=695.43 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=13 q_lam=443.57 kW/m^2 q_turb=699.39 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=14 q_lam=422.72 kW/m^2 q_turb=696.00 kW/m^2 alt=6096.00 m t=0.00 s
i=0 station=15 q_lam=405.43 kW/m^2 q_turb=693.29 kW/m^2 alt=6096.00 m t=0.00 s
'''