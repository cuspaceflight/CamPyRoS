"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
Nomenclature is the mostly same as used in https://apps.dtic.mil/sti/pdfs/AD0642855.pdf

COORDINATE SYSTEM NOMENCLATURE
x,y,z = Body coordinate system (origin on rocket, rotates with the rocket)
X,Y,Z = Inertial coordinate system (does not rotate, fixed origin relative to centre of the Earth)
X', Y', Z' = Launch site coordinate system (origin on launch site, rotates with the Earth)


Would be useful to define directions, e.g. maybe
- Body:
    x in direction the rocket points - y and z aligned with launch site coordinates at t=0

- Launch site:
    X' points East, Y' points North, Z' points upwards (towards space)
    
- Inertial:
    Aligned with launch site coordinate system at t=0

'''

import csv
import numpy as np
import matplotlib.pyplot as plt


#Class to store atmospheric model data
class Atmosphere:
    def __init__(self, adat, ddat, sdat, padat): 
        self.adat = adat    #altitudes
        self.ddat = ddat    #densities
        self.sdat = sdat    #speed of sound
        self.padat = padat  #pressures

#Import standard atmosphere - copied from Joe Hunt's simulation
with open('atmosphere_data.csv') as csvfile:
    standard_atmo_data = csv.reader(csvfile)
    adat, ddat, sdat, padat = [], [], [], []
    next(standard_atmo_data)
    for row in standard_atmo_data:
        adat.append(float(row[0]))
        ddat.append(float(row[1]))
        sdat.append(float(row[2]))
        padat.append(float(row[3]))

StandardAtmosphere = Atmosphere(adat,ddat,sdat,padat)       #Built-in Standard Atmosphere data that you can use


#Class to store the data on a hybrid motor
class HybridMotor:
    def __init__(self, motor_time_data, prop_mass_data, cham_pres_data, throat_data,
                 gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data): 
        self.motor_time_data = motor_time_data
        self.prop_mass_data = prop_mass_data
        self.cham_pres_data = cham_pres_data
        self.throat_data = throat_data
        self.gamma_data = gamma_data
        self.nozzle_efficiency_data = nozzle_efficiency_data
        self.exit_pres_data = exit_pres_data
        self.area_ratio_data = area_ratio_data


#Class to store data on your launch site
class LaunchSite:
  def __init__(self, rail_length, alt, long, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
    self.rail_length = rail_length
    self.alt = alt                  #Altitude
    self.long = long                #Longitude
    self.lat = lat                  #Latitude
    self.wind = wind                #Wind speed vector relative to the surface of the Earth, [X', Y', Z'] m/s
    self.atmosphere = atmosphere    #An Atmosphere object to get atmosphere data from
    
    
#Class to store all the import information on a rocket
class Rocket:
  def __init__(self, dry_mass, Ixx, Iyy, Izz, motor, aerodynamic_data, launch_site):
    '''
    dry_mass - Mass without fuel
    Ixx
    Iyy
    Izz - principal moments of inertia - rocket points in the xx direction
    motor - some kind of Motor object - currently the only option is HybridMotor
    aerodynamic_data - non-functional at the moment
    launch_site - a LaunchSite object
    '''
      
    self.launch_site = launch_site                  #LaunchSite object
      
    self.dry_mass = dry_mass                    #Dry mass kg
    self.Ixx = Ixx                              #Principal moments of inertia kg m2
    self.Iyy = Iyy
    self.Izz = Izz
    
    self.motor = motor                              #Motor object containing motor data
    self.aerodynamic_data = aerodynamic_data        #e.g. drag coefficients


    self.time = 0               #Time since ignition s
    self.m = 0                  #Instantaneous mass (will vary as fuel is used) kg
    self.w = [0,0,0]            #Angular velocity of the x,y,z coordinate system in the X,Y,Z coordinate system [X,Y,Z] rad/s
    self.v = [0,0,0]            #Velocity in intertial coordinates [X, Y, Z] m/s
    self.pos = [0,0,0]          #Position in inertial coordinates [X, Y, Z] m
    self.alt = launch_site.alt  #Altitude
    
    def body_to_intertial(vector):  #Convert a vector in x,y,z to X,Y,Z
        pass
    
    def surfacevelocity_to_inertial(vector):    #Converts a surface velocity to an inertial one
        print("surfacevelocity_to_inertial is not yet functional")
        return [0,0,0]                          #Doesn't work yet
    
    def aero_forces():          #Returns aerodynamic forces and moments based on current pos and vel 
        v_rel_wind = self.v - self.surfacevelocity_to_intertial(self.launch_site.wind)
        V_a = np.linalg.norm(v_rel_wind)
        alpha = np.arctan(v_rel_wind[2]/v_rel_wind[0])
        beta = np.arctan(v_rel_wind[1] / (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.arctan( (v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5 / v_rel_wind[0])
        
        alpha_star = np.arctan(v_rel_wind[2] / (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )
        beta_star = np.arctan(v_rel_wind[1]/v_rel_wind[0])
            
        #Dynamic pressure
        q = 0.5*self.launch_site.atmosphere.ddat[0]  
        
        #Characteristic area and moment arm
        S = 0
        d = 0
        
        #Drag/Force coefficients - note in the NASA document they use the notation angular velocity w = [p, q, r] rad/s
        Cx = 0
        Cna = 0
        Cl = 0
        Clo = 0
        Cma = 0
        Cmr = 0
        Cmq = 0
        
        #Forces
        Fx = Cx*q*S
        Fy = -Cna*np.sin(beta)*q*S
        Fz = -Cna*np.sin(alpha_star)*q*S
        
        #Moments
        L = (Clo + Cl*(self.w[0]*d / (2*V_a)) ) *q*S*d
        M = (Cma*np.sin(alpha_star) + Cmq*(self.w[1]*d / (2*V_a) ) ) *q*S*d
        N = (-Cma*np.sin(beta) + Cmr*(self.w[2]*d / (2*V_a) )) *q*S*d
        
        return [Fx,Fy,Fz], [L,M,N]
        
    def motor_forces():         #Returns thrust and moments generated by the motor, based on current conditions
        pass
    
    
    def accelerations(forces, moments):     #Returns translational and rotational accelerations on the rocket, given the applied forces
        pass                                #Some kind of implementation of the equations of motion


#Functions
def integrate(initial_conditions, accelerations):      #Not sure what other inputs would be needed
    pass


def run_simulation(rocket):     #'rocket' can be a Rocket object
    pass                        #Maybe returns a dictionary of data?



def plot_altitude_time(simulation_output):  #takes data from a simulation and plots nice graphs for you
    pass        
