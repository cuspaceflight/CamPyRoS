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


#Class to store data on your launch site
class LaunchSite:
  def __init__(self, rail_length, rail_azimuth, rail_polar ,alt, longi, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
    self.rail_length = rail_length
    self.rail_azimuth = rail_azimuth #Angle that rail points from straight up
    self.rail_polar = rail_polar    #Angle from north that rail inclination points in
    self.alt = alt                  #Altitude
    self.longi = longi                #Longitude
    self.lat = lat                  #Latitude
    self.wind = wind                #Wind speed vector relative to the surface of the Earth, [x_] m/s
    self.atmosphere = atmosphere    #An Atmosphere object to get atmosphere data from
    
    
#Class to store all the import information on a rocket
class Rocket:
    def __init__(self, dry_mass, Ixx, Iyy, Izz, motor, aerodynamic_data, launch_site, h, variable):
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
        self.h = h            #Time step size (can evolve)
        self.variable_time = variable   #Vary timestep with error (option for ease of debugging)
        self.m = 0                  #Instantaneous mass (will vary as fuel is used) kg
        self.orientation = np.array([0,0,0]) #yaw pitch roll  of the body frame in the inertial frame rad
        self.w = np.array([0,0,0])            #rate of change of yaw pitch roll rad/s
        self.pos = pos_launch_to_inertial(np.array([0,0,0]),launch_site,0)         #Position in inertial coordinates [x,y,z] m
        self.v = np.array([0,0,0])#vel_launch_to_inertial(np.array([0,0,0]),launch_site,0)           #Velocity in intertial coordinates [x',y',z'] m/s
        self.alt = launch_site.alt  #Altitude
        self.on_rail=True
    
    def body_to_intertial(self,vector):  #Convert a vector in x,y,z to X,Y,Z
        return np.matmul(rot_matrix(self.orientation),vector)
    
    def aero_forces(self):          #Returns aerodynamic forces and moments based on current pos and vel 
        pass
        
    def motor_forces(self):         #Returns thrust and moments generated by the motor, based on current conditions
        pass

    def gravity(position):      
        return 9.81
    
    def altitude(position):
        return np.linalg.norm(position)-6317000
    
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
        r=np.stack((self.pos,self.orientation))+b[0]*l_1+b[1]*l_2+b[2]*l_3+b[3]*l_4+b[4]*l_5+b[5]*l_6 #+O(h^6) This is movement of the body frame from wherever it is- needs to be transformed to inertial frame

        if(self.variable_time==True):
            v_=np.stack((self.v,self.w))+b_[0]*k_1+b_[1]*k_2+b_[2]*k_3+b_[3]*k_4+b_[4]*k_5+b_[5]*k_6 #+O(h^5)
            r_=np.stack((self.pos,self.orientation))+b_[0]*l_1+b_[1]*l_2+b_[2]*l_3+b_[3]*l_4+b_[4]*l_5+b_[5]*l_6 #+O(h^5)

            scale_v=atol_v+rtol_v*abs(np.maximum.reduce([v,np.stack((self.v,self.w))]))
            scale_r=atol_r+rtol_r*abs(np.maximum.reduce([r,np.stack((self.pos,self.orientation))))
            err_v=(v-v_)/scale_v
            err_r=(r-r_)/scale_r
            err=max(err_v,err_r)
            self.h=sf*self.h*pow(max(err),-1/5)

        self.v=v[0]
        self.w=v[1]
        self.pos=r[0]
        self.orientation=r[1]
        
def pos_launch_to_inertial(position,launch_site,time):#takes position vector and launch site object
    #Adapted from https://gist.github.com/mpkuse/d7e2f29952b507921e3da2d7a26d1d93 
    phi = launch_site.lat / 180. * np.pi
    lambada = (launch_site.longi+time*7.292115e-5) / 180. * np.pi
    h = launch_site.alt

    e = 0.081819191 #earth ecentricity
    q = np.sin( phi )
    N = 6378137.0 / np.sqrt( 1 - e*e * q*q )
    X = (N + h) * np.cos( phi ) * np.cos( lambada )
    Y = (N + h) * np.cos( phi ) * np.sin( lambada )
    Z = (N*(1-e*e) + h) * np.sin( phi )
    return np.array([X,Y,Z])

def vel_launch_to_inertial(launch_site):
    #This assums spherical earth and as such will need updating if we go for a more complicated model
    pass

def rot_matrix(orientation,inverse=False):#left hand multiply this (i.e. np.matmul(rotation_matrix(....),vec)) 
    #can be used on accelerations in the body frame for passing to the integrator, don't think the rate of angular velocity needs changing
    if inverse==True:
        orientation=[-inv for inv in orientation]
    r_x=np.array([[1,0,0],
        [0,np.cos(orientation[2]),-np.sin(orientation[2])],
        [0,np.sin(orientation[2]),np.cos(orientation[2])]])
    r_y=np.array([[np.cos(orientation[1]),0,np.sin(orientation[1])],
        [0,1,0],
        [-np.sin(orientation[1]),0,np.cos(orientation[1])]])
    r_z=np.array([[np.cos(orientation[0]),-np.sin(orientation[0]),0],
        [np.sin(orientation[0]),np.cos(orientation[0]),0],
        [0,0,1]])
    if inverse==True:
        rot = np.matmul(r_z.transpose(),np.matmul(r_y.transpose(),r_x.transpose())).transpose()
    else:
        rot = np.matmul(r_z,np.matmul(r_y,r_x))
    return rot

def run_simulation(rocket):     #'rocket' can be a Rocket object
    record={"position":[],"velocity":[],"orientation":[],"mass":[]}#all in inertial frame
    while rocket.alt>0:
        rocket.step()
        record["position"].append(pos_inertial_to_launch(rocket.pos,rocket.launch_site,rocket.time))
        record["velocity"].append(vel_inertial_to_launch(rocket.vel,rocket.launch_site,rocket.time))
        record["orientation"].append(orient_inertial_to_launch(rocket.point,rocket.launch_site,rocket.time))
        record["mass"].append(rocket.m)

    return 0

def plot_altitude_time(simulation_output):  #takes data from a simulation and plots nice graphs for you
    pass        
