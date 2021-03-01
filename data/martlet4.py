import CamPyRoS as trajectory
import csv
import time
import numpy as np
import pandas as pd

"""Import motor data to use for the mass model"""
motor_csv = pd.read_csv("novus_sim_6.1/motor_out.csv")

time_array = motor_csv["Time"]
smass_array = motor_csv["Solid Fuel Mass (kg)"]
S_DEN = motor_csv["Solid Fuel Density (kg/m^3)"][0]
S_L = motor_csv["Solid Fuel Length (m)"][0]
S_ROUT = motor_csv["Solid Fuel Outer Diameter (m)"][0]
vmass_array = motor_csv["Vapour Mass (kg)"]
vden_array = motor_csv["Vapour Density (kg/m^3)"]
lmass_array = motor_csv["Liquid Mass (kg)"]
lden_array = motor_csv["Liquid Density (kg/m^3)"]

"""Rocket parameters"""
DRY_MASS = 60  # Rocket dry mass (kg)
ROCKET_L = 6.529  # Rocket length (m)
ROCKET_R = 98.5e-3  # Rocket radius (m)
ROCKET_T = 1e-2  # Rocket wall thickness (m) - used when approximating the rocket airframe as a thin walled cylinder
POS_TANK_BOTTOM = (
    4.456  # Distance between the nose tip and the bottom of the nitrous tank (m)
)
POS_SOLIDFUEL_BOTTOM = (
    4.856 + S_L
)  # Distance between the nose tip and bottom of the solid fuel grain (m)
REF_AREA = 0.0305128422  # Reference area for aerodynamic coefficients (m^2)

"""Set up aerodynamic properties"""
# Get approximate values for the rotational damping coefficients
C_DAMP_PITCH = trajectory.pitch_damping_coefficient(
    ROCKET_L, ROCKET_R, fin_number=4, area_per_fin=0.07369928
)
C_DAMP_ROLL = 0

# Import drag coefficients from RASAero II
aero_data = trajectory.AeroData.from_rasaero(
    "data/Martlet4RASAeroII.CSV", REF_AREA, C_DAMP_PITCH, C_DAMP_ROLL
)
# aero_data.show_plot()   #Show plots of how the program interpreted the data, so you can visually check if it's correct

"""Set up the mass model"""
mass_model = trajectory.MassModel()
mass_model.add_hollowcylinder(
    DRY_MASS, ROCKET_R, ROCKET_R - ROCKET_T, ROCKET_L, ROCKET_L / 2
)
mass_model.add_liquidtank(
    lmass_array,
    lden_array,
    time_array,
    ROCKET_R,
    POS_TANK_BOTTOM,
    vmass_array,
    vden_array,
)
mass_model.add_solidfuel(
    smass_array, time_array, S_DEN, S_ROUT, S_L, POS_SOLIDFUEL_BOTTOM
)

"""Create the other objects needed to initialise the Rocket object"""
pulsar = trajectory.Motor.from_novus("novus_sim_6.1/motor_out.csv", pos=ROCKET_L)

"""
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
"""
launch_site = trajectory.LaunchSite(
    rail_length=5,
    rail_yaw=0,
    rail_pitch=0,
    alt=1,
    longi=0,
    lat=0,
    variable_wind=False,
    default_wind=np.array([5, 0, 0]),
)  # Use this version if you don't want to use the real wind (e.g. to test something else)

parachute = trajectory.Parachute(
    main_s=13.9,
    main_c_d=0.78,
    drogue_s=1.13,
    drogue_c_d=0.78,
    main_alt=1000,
    attach_distance=0,
)

"""Create the Rocket object"""
martlet4 = trajectory.Rocket(
    mass_model,
    pulsar,
    aero_data,
    launch_site,
    h=0.05,
    variable=True,
    alt_poll_interval=1,
    parachute=parachute,
)
