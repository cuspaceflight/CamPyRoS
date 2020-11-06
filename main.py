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
import scipy.interpolate

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

#Class used to get the data we need from a RasAero II 'Aero Plots' export file
class RasAeroData:
        def __init__(self, file_location_string, area = 0.0305128422): 
            '''
            file_location_string - the directory that the RasAeroII .CSV file is, containing the aero data
            A - the area that was used to normalise the coefficients, in m^2
            '''
            self.area = area
            
            with open(file_location_string) as csvfile:
                aero_data = csv.reader(csvfile)
            
                Mach_raw = []
                alpha_raw = []
                CA_raw = []
                COP_raw = []
                CN_raw = []
    
                #Extract the raw data from the .csv file
                next(aero_data)            
                for row in aero_data:
                    Mach_raw.append(float(row[0]))
                    alpha_raw.append(float(row[1]))
                    CA_raw.append(float(row[5]))
                    COP_raw.append(float(row[12]))
                    CN_raw.append(float(row[8]))
            
            #Seperate the data by angle of attack.
            Mach = []
            CA_0 = []  #CA at alpha = 0
            CA_2 = []  #CA at alpha = 2
            CA_4 = []  #CA at alpha = 4
            COP_0 = []
            COP_2 = []
            COP_4 = []
            CN_0 = []
            CN_2 = []
            CN_4 = []
                
            for i in range(len(Mach_raw)):
                if alpha_raw[i] == 0:
                    Mach.append(Mach_raw[i])
                    CA_0.append(CA_raw[i])
                    COP_0.append(COP_raw[i])
                    CN_0.append(CN_raw[i])
                
                elif alpha_raw[i] == 2:
                    CA_2.append(CA_raw[i])
                    COP_2.append(COP_raw[i])
                    CN_2.append(CN_raw[i])    
                
                elif alpha_raw[i] == 4:
                    CA_4.append(CA_raw[i])
                    COP_4.append(COP_raw[i])
                    CN_4.append(CN_raw[i])   
            
            #Make sure all the lists are the same length - this is needed because it seems the alpha=4 data only has 2499 points, but the others have 2500
            CA_0, CA_2, CA_4 = CA_0[:2498], CA_2[:2498], CA_4[:2498]
            CN_0, CN_2, CN_4 = CN_0[:2498], CN_2[:2498], CN_4[:2498]
            COP_0, COP_2, COP_4 = COP_0[:2498], COP_2[:2498], COP_4[:2498]
            Mach = Mach[:2498]
            
            #Generate grids of the data
            CA = np.array([CA_0, CA_2, CA_4])
            CN = np.array([CN_0, CN_2, CN_4])
            COP = 0.0254*np.array([COP_0, COP_2, COP_4])    #Convert inches to m
            alpha = [0,2,4]
                    
            #Generate functions (note these are funcitons, not variables) which return a coefficient given (Mach, alpha)
            self.COP = scipy.interpolate.interp2d(Mach, alpha, COP)
            self.CA = scipy.interpolate.interp2d(Mach, alpha, CA)
            self.CN = scipy.interpolate.interp2d(Mach, alpha, CN)
            
            
#Class to store data on your launch site
class LaunchSite:
  def __init__(self, rail_length, alt, long, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
    self.rail_length = rail_length
    self.alt = alt                  #Altitude
    self.long = long                #Longitude
    self.lat = lat                  #Latitude
    self.wind = np.array(wind)      #Wind speed vector relative to the surface of the Earth, [X', Y', Z'] m/s
    self.atmosphere = atmosphere    #An Atmosphere object to get atmosphere data from
    
    
#Class to store all the import information on a rocket
class Rocket:
    def __init__(self, dry_mass, Ixx, Iyy, Izz, motor, aero, launch_site):
        '''
        dry_mass - Mass without fuel
        Ixx
        Iyy
        Izz - principal moments of inertia - rocket points in the xx direction
        motor - some kind of Motor object - currently the only option is HybridMotor
        aero - Aerodynamic coefficients and stability derivatives
        launch_site - a LaunchSite object
        '''
          
        self.launch_site = launch_site                  #LaunchSite object
          
        self.dry_mass = dry_mass                    #Dry mass kg
        self.Ixx = Ixx                              #Principal moments of inertia kg m2
        self.Iyy = Iyy
        self.Izz = Izz
        
        self.motor = motor                              #Motor object containing motor data
        self.aero = aero        #e.g. drag coefficients
    
    
        self.time = 0               #Time since ignition s
        self.m = 0                  #Instantaneous mass (will vary as fuel is used) kg
        self.w = np.array([0,0,0])  #Angular velocity of the x,y,z coordinate system in the X,Y,Z coordinate system [X,Y,Z] rad/s
        self.v_b = np.array([0,0,0])#Velocity in body coordinates [x, y, z] m/s
        self.pos = np.array([0,0,0])#Position in inertial coordinates [X, Y, Z] m
        self.alt = launch_site.alt  #Altitude
    
    def body_to_intertial(self, vector):  #Convert a vector in x,y,z to X,Y,Z
        pass
    
    def surfacevelocity_to_inertial(self, vector):    #Converts a surface velocity to an inertial one
        print("surfacevelocity_to_inertial is not yet functional")
        return [0,0,0]                          #Doesn't work yet
    
    def aero_forces(self):
        '''
        Returns aerodynamic forces, and the point at which they act (i.e. the centre of pressure) 
        
        Note that this currently ignores the damping moment generated by the rocket is rotating about its long axis
        '''
        
        #Velocities and Mach number
        v_rel_wind = self.v_b - self.surfacevelocity_to_inertial(self.launch_site.wind)
        v_a = np.linalg.norm(v_rel_wind)
        v_sound = np.interp(self.alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.sdat)
        mach = v_a/v_sound
        
        #Angles
        alpha = np.arctan(v_rel_wind[2]/v_rel_wind[0])
        beta = np.arctan(v_rel_wind[1] / (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.arctan( (v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5 / v_rel_wind[0])
        alpha_star = np.arctan(v_rel_wind[2] / (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )
        beta_star = np.arctan(v_rel_wind[1]/v_rel_wind[0])
            
        #Dynamic pressure at the current altitude and velocity - WARNING: Am I using the right density?
        q = 0.5*np.interp(self.alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.ddat)*(v_a**2)
        
        #Characteristic area
        S = self.aero.area
        
        #Drag/Force coefficients
        Cx = self.aero.CA(mach, abs(delta))         #WARNING: Not sure if I'm using the right angles for these all
        Cz = self.aero.CN(mach, abs(alpha_star))    #OR IF THIS IS THE CORRECT WAY TO USE CN
        Cy = self.aero.CN(mach, abs(beta)) 
        
        #Forces
        Fx = -np.sign(v_rel_wind)[0]*Cx*q*S                         
        Fy = -np.sign(v_rel_wind)[1]*Cy*q*S                         
        Fz = -np.sign(v_rel_wind)[2]*Cz*q*S
        
        #Position where moments act:
        COP = self.aero.COP(mach, abs(delta))[0]

        return np.array([Fx,Fy,Fz]), COP
        
    def motor_forces(self):         #Returns thrust and moments generated by the motor, based on current conditions
        pass
    
    
    def accelerations(self, forces, moments):     #Returns translational and rotational accelerations on the rocket, given the applied forces
        pass                                #Some kind of implementation of the equations of motion


#Functions
def integrate(initial_conditions, accelerations):      #Not sure what other inputs would be needed
    pass


def run_simulation(rocket):     #'rocket' can be a Rocket object
    pass                        #Maybe returns a dictionary of data?



def plot_altitude_time(simulation_output):  #takes data from a simulation and plots nice graphs for you
    pass        
