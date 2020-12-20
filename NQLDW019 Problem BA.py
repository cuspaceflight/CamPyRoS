#To check data against page 70 of the NASA "TANGENT OGIVE NOSE AERODYNAMIC HEATING PROGRAM - NQLDW019 (NASA)" documentation
import trajectory, trajectory.aero, csv
import trajectory.post as post
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

'''Specify the nosecone'''
#xprime = 2.741 ft = 0.8354568 m
#yprime = 0.5 ft = 0.1524 m 
#tangent_ogive = trajectory.post.TangentOgive(xprime = 0.8354568, yprime = 0.1524)
tangent_ogive = post.TangentOgive(xprime = 2.741, yprime = 0.5)

for i in range(15):
    print("R({}) = {} S({}) = {}".format(i+1, tangent_ogive.r(i+1), i+1, tangent_ogive.S(i+1)))

#Freestream conditions:
ALT = 15240         #50000 ft
VINF = 1219.2       #4000 ft/s
ALPHA = 10 * np.pi/180

#Properties using standard atmosphere
TINF = 216.650              #K
PINF = 11597.3              #Pa
RHOINF = 0.186481           #kg/m3
speed_of_sound = 295.070    #m/s
MINF = VINF/speed_of_sound

#Shockwave at point 1
THETA1 = 30.675 *np.pi/180  #rad, obtained from the FORTAN output
oblique_shock_data = post.oblique_shock(THETA1, MINF, TINF, PINF, RHOINF)
cone_shock_data = post.oblique_shock(tangent_ogive.theta + ALPHA, MINF, TINF, PINF, RHOINF)

print("MINF = {} THETA1 = {} rad tangent_ogive.theta + ALPHA = {} rad".format(MINF, THETA1, tangent_ogive.theta + ALPHA))
print("From NASA report: TS(1) = 470.289 K")
print("From Python oblique_shock(): TS = {} K".format(oblique_shock_data[3]))
print("From Python cone_shock(): TS = {} K".format(cone_shock_data[3]))

