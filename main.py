"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
Nomenclature is the mostly same as used in https://apps.dtic.mil/sti/pdfs/AD0642855.pdf

COORDINATE SYSTEM NOMENCLATURE
x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
x_i,y_i,z_i = Inertial coordinate system (does not rotate, fixed origin relative to centre of the Earth)
x_l, y_l, z_l = Launch site coordinate system (origin on launch site, rotates with the Earth)


Would be useful to define directions, e.g. maybe
- Body:
    x_b in direction the rocket points - y_b and z_b aligned with launch site coordinates at t=0

- Launch site:
    x_l points East, y_l points North, z_l points upwards (towards space)
    
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

#RK4 parameters
c=[0,1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0,1.0,1.0]
a=[[0,               0,              0,               0,            0,              0        ],
[1.0/5.0,         0,              0,               0,            0,              0        ],
[3.0/40.0,        9.0/40.0,       0,               0,            0,              0        ],
[44.0/45.0,      -56.0/15.0,      32.0/9.0,        0,            0,              0        ],
[19327.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,  0,              0        ],
[9017.0/3168.0,  -355.0/33.0,     46732.0/5247.0,  49.0/176.0,  -5103.0/18656.0, 0        ],
[35.0/384.0,      0,              500.0/1113.0,    125.0/192.0, -2187.0/6784.0,  11.0/84.0]]
b=[35.0/384.0,  0,  500.0/1113.0, 125.0/192.0, -2187.0/6784.0,  11.0/84.0,  0]
b_=[5179.0/57600.0,  0,  7571.0/16695.0,  393.0/640.0,  -92097.0/339200.0, 187.0/2100.0,  1.0/40.0]

atol_v=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #absolute error of each component of v and w
rtol_v=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #relative error of each component of v and w
atol_r=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #absolute error of each component of position and pointing
rtol_r=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #relative error of each component of position and pointing
sf=0.98 #Safety factor for h scaling

#Class to store the data on a hybrid motor
class Motor:
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
  def __init__(self, rail_length, rail_azimuth, rail_polar ,alt, longi, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
    self.rail_length = rail_length
    self.rail_azimuth = rail_azimuth #Angle that rail points from straight up
    self.rail_polar = rail_polar    #Angle from north that rail inclination points in
    self.alt = alt                  #Altitude
    self.longi = longi                #Longitude
    self.lat = lat                  #Latitude
    self.wind = np.array(wind)      #Wind speed vector relative to the surface of the Earth, [x_l, y_l, z_l] m/s
    self.atmosphere = atmosphere    #An Atmosphere object to get atmosphere data from
    
    
#Class to store all the import information on a rocket
class Rocket:
    def __init__(self, dry_mass, Ixx, Iyy, Izz, motor, aero, launch_site, h, variable):
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
    
        self.time = 0                                                       #Time since ignition s
        self.h = h                                                          #Time step size (can evolve)
        self.variable_time = variable                                       #Vary timestep with error (option for ease of debugging)
        self.m = 0                                                          #Instantaneous mass (will vary as fuel is used) kg
        self.w = np.array([0,0,0])                                          #Angular velocity of the x,y,z coordinate system in the X,Y,Z coordinate system - the euler angles as given in the diagram in readme [alpha',beta',gamma'] with the body frame as the red rad/s
        self.v = vel_launch_to_inertial(np.array([0,0,0]),launch_site,0)    #Velocity in intertial coordinates [x',y',z'] m/s
        self.v_b = np.array([0,0,0])                                        #Velocity in body coordinates [x_b, y_b, z_b] m/s
        self.pos = pos_launch_to_inertial(np.array([0,0,0]),launch_site,0)  #Position in inertial coordinates [x,y,z] m
        self.alt = launch_site.alt                                          #Altitude
        self.point = np.array([0,0,0])                                      #Euler angles given by readme diagram with red as body frame [alpha,beta,gamma] rad
        self.on_rail=True
        
    def surfacevelocity_to_inertial(self, vector):                  #Converts a surface velocity to an inertial one
        print("surfacevelocity_to_inertial is not yet functional")  #Doesn't work yet
        return [0,0,0]                                          
    
    def aero_forces(self):
        '''
        Returns aerodynamic forces, and the point at which they act (i.e. the centre of pressure) 
        
        Note that:
         -This currently ignores the damping moment generated by the rocket is rotating about its long axis
         -I'm not sure if I used the right angles for calculating CN as the angles of attack vary
             (same for CA as angles of attack vary)
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
        Cz = self.aero.CN(mach, abs(alpha_star))    #Or if this is the correct way to use CN
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

    def gravity(position):
        return 9.81
    
    def acceleration(self,pos,v_b,w,time,alt):     #Returns translational and rotational accelerations on the rocket, given the applied forces
        #Some kind of implementation of the equations of motion
        #Out put needs to be relative to the inertial frame (doesn't really mean anything for it to be accelerating in the body frame but makes sense for forces to be applied in body frame)
        out = np.stack((np.array([0,0,0]),np.array([0,0,0])))
        return out
    
    def step(self):
        #Implimenting a time step variable method based on Numerical recipes (link in readme)
        #Including possibility to update altitude incase it is decided that the altitude change between steps effects air pressure significantly but will keep constant for now 
        k_1=self.h*self.acceleration(self.pos,self.v,self.w,self.time,self.alt)
        k_2=self.h*self.acceleration(self.pos,self.v+a[1][0]*k_1,self.w,self.time+c[1]*self.h,self.alt)
        k_3=self.h*self.acceleration(self.pos,self.v+a[2][0]*k_1+a[2][1]*k_2,self.w,self.time+c[2]*self.h,self.alt)
        k_4=self.h*self.acceleration(self.pos,self.v+a[3][0]*k_1+a[3][1]*k_2+a[3][2]*k_3,self.w,self.time+c[3]*self.h,self.alt)
        k_5=self.h*self.acceleration(self.pos,self.v+a[4][0]*k_1+a[4][1]*k_2+a[4][2]*k_3+a[4][3]*k_4,self.w,self.time+c[4]*self.h,self.alt)
        k_6=self.h*self.acceleration(self.pos,self.v+a[5][0]*k_1+a[5][1]*k_2+a[5][2]*k_3+a[5][3]*k_4+a[5][4]*k_5,self.w,self.time+c[4]*self.h,self.alt)
        
        l_1=np.stack((self.v,self.w))
        l_2=l_1+a[1][0]*k_1
        l_3=l_1+a[2][0]*k_1+a[2][1]*k_2
        l_4=l_1+a[3][0]*k_1+a[3][1]*k_2+a[3][2]*k_3
        l_5=l_1+a[4][0]*k_1+a[4][1]*k_2+a[4][2]*k_3+a[4][3]*k_4
        l_6=l_1+a[5][0]*k_1+a[5][1]*k_2+a[5][2]*k_3+a[5][3]*k_4+a[5][4]*k_5

        v=np.stack((self.v,self.w))+b[0]*k_1+b[1]*k_2+b[2]*k_3+b[3]*k_4+b[4]*k_5+b[5]*k_6 #+O(h^6)
        r=np.stack((self.pos,self.point))+b[0]*l_1+b[1]*l_2+b[2]*l_3+b[3]*l_4+b[4]*l_5+b[5]*l_6 #+O(h^6) This is movement of the body frame from wherever it is- needs to be transformed to inertial frame

        if(self.variable_time==True):
            v_=np.stack((self.v,self.w))+b_[0]*k_1+b_[1]*k_2+b_[2]*k_3+b_[3]*k_4+b_[4]*k_5+b_[5]*k_6 #+O(h^5)
            r_=np.stack((self.pos,self.point))+b_[0]*l_1+b_[1]*l_2+b_[2]*l_3+b_[3]*l_4+b_[4]*l_5+b_[5]*l_6 #+O(h^5)

            scale_v=atol_v+rtol_v*abs(np.maximum.reduce([v,self.v_b]))
            scale_r=atol_r+rtol_r*abs(np.maximum.reduce([r,self.r_b]))
            err_v=(v-v_)/scale_v
            err_r=(v-v_)/scale_r
            err=max(err_v,err_r)
            self.h=sf*self.h*pow(max(err),-1/5)

        self.v=v[0]
        self.w=v[1]
        self.pos=r[0]
        self.point=r[1]

def pos_launch_to_inertial(position,launch_site,time): 
    '''
    Takes position vector and launch site object
    Adapted from https://gist.github.com/mpkuse/d7e2f29952b507921e3da2d7a26d1d93 
    '''
    phi = launch_site.lati / 180. * np.pi
    lambada = (launch_site.longi+time*7.292115e-5) / 180. * np.pi
    h = launch_site.alt

    e = 0.081819191 #earth ecentricity
    q = np.sin( phi )
    N = 6378137.0 / np.sqrt( 1 - e*e * q*q )
    X = (N + h) * np.cos( phi ) * np.cos( lambada )
    Y = (N + h) * np.cos( phi ) * np.sin( lambada )
    Z = (N*(1-e*e) + h) * np.sin( phi )
    return np.array([X,Y,Z])

def vel_launch_to_inertial(velocity,launch_site,time):
    pass

def orient_inertial_to_launch(orientation, launch_site, time):
    pass

def accel_body_to_inertial(acceleration, position, time):
    pass

def orient_body_to_inertial(orientation, position, time):
    pass


def run_simulation(rocket):     #'rocket' can be a Rocket object
    record={"position":[],"velocity":[],"orientation":[],"mass":[]}#all in inertial frame
    while rocket.alt>0:
        rocket.step()
        record["position"].append(pos_inertial_to_launch(rocket.pos,rocket.launch_site,rocket.time))
        record["velocity"].append(vel_inertial_to_launch(rocket.vel,rocket.launch_site,rocket.time))
        record["orientation"].append(orient_inertial_to_launch(rocket.point,rocket.launch_site,rocket.time))
        record["mass"].append(rocket.m)


def plot_altitude_time(simulation_output):  #takes data from a simulation and plots nice graphs for you
    pass        
