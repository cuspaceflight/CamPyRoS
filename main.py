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
    y points east and z north at take off (before rail alignment is accounted for)

- Launch site:
    z points perpendicular to the earth, y in the east direction and x tangential to the earth pointing south
    
- Inertial:
    Aligned with launch site coordinate system at t=0
    I have been thinking about it as Z points to north from centre of earth, x aligned with launchsite at start and y orthogonal

'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd


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

#These should be moved to the rocket class or at least the run simulation function
atol_v=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #absolute error of each component of v and w
rtol_v=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #relative error of each component of v and w
atol_r=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #absolute error of each component of position and pointing
rtol_r=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]) #relative error of each component of position and pointing
sf=0.98 #Safety factor for h scaling

r_earth = 6378137 #(earth's semimarjor axis in meters)
#e_earth = 0.081819191 #earth ecentricity
e_earth = 0 #for simplicity of other calculations for now - if changed need to update the launchsite orientation and velocity transforms
ang_vel_earth=7.292115090e-5

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
  def __init__(self, rail_length, rail_yaw, rail_pitch,alt, longi, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
    self.rail_length = rail_length
    self.rail_yaw = rail_yaw        #Angle of rotation about the z axis (north pointing)
    self.rail_pitch = rail_pitch    #Angle of rotation about "East" pointing y axis - in order to simplify calculations below this needs to be measured in the yaw then pitch order
    self.alt = alt                  #Altitude
    self.longi = longi              #Longitude
    self.lat = lat                  #Latitude
    self.wind = np.array(wind)      #Wind speed vector relative to the surface of the Earth, [x_l, y_l, z_l] m/s
    self.atmosphere = atmosphere    #An Atmosphere object to get atmosphere data from
    
#Class to use for the variable mass model for the rocket
class CylindricalMassModel:
    def __init__(self, mass, time, l, r):
        '''
        Assumes the rocket is a solid cylinder, constant volume, which has a mass that reduces with time (i.e. the density of the cylinder reducs)
        
        mass = A list of masses, at each moment in time (after ignition)
        time = A list of times, corresponding to the masses in the 'mass' list
        l = length of the cylinder
        r = radius of the cylinder
        
        self.mass(time) - Returns the mass at a 'time' after ignition
        self.ixx(time), self.iyy(time), self.izz(time) - Each returns the principa moment of inertia at a 'time' after ignition
        self.cog(time) - Returns centre of gravity, as a distance from the nose tip. Would be a constant, 'time' does not actually affect it (but is included for completeness)
        '''
        self.mass_interp = scipy.interpolate.interp1d(time, mass)
        self.time = time
        self.l = l
        self.r = r

    def mass(self, time):
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            print("Trying to calculate mass or moment of inertia data before first time datapoint - rounding from time={} to time={}".format(time, self.time[0]))
            return self.mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.mass_interp(time)
        else:
            return self.mass_interp(self.time[-1])

    def ixx(self, time):
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            print("Trying to calculate mass or moment of inertia data before first time datapoint - rounding from time={} to time={}".format(time, self.time[0]))
            return (1/2)* self.r**2 * self.mass(self.time[0])
        elif time < self.time[-1]:
            return (1/2)* self.r**2 * self.mass(time)
        else:
            return (1/2)* self.r**2 * self.mass(self.time[-1])

      
    def iyy(self, time):
        if time < 0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            print("Trying to calculate mass or moment of inertia data before first time datapoint - rounding from time={} to time={}".format(time, self.time[0]))
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[0])
        elif time < self.time[-1]:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(time)
        else:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[-1])
      
    def izz(self, time):
        return self.iyy(time)
    
    def cog(self, time):
        return self.l/2


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
            
            
#Class to store all the import information on a rocket
class Rocket:
    def __init__(self, mass_model, motor, aero, launch_site, h, variable):
        '''
        motor - some kind of Motor object - currently the only option is HybridMotor
        mass_model - The object used to model the rocket's mass - currently the only option is CylindricalMassModel
        aero - Aerodynamic coefficients and stability derivatives
        launch_site - a LaunchSite object
        '''

        self.launch_site = launch_site                  #LaunchSite object
        self.motor = motor                              #Motor object containing motor data
        self.aero = aero                                #object containing aerodynamic information
        self.mass_model = mass_model                    #Object used to model the mass of the rocket
        
        self.time = 0                           #Time since ignition (seconds)
        self.h = h                              #Time step size (can evolve)
        self.variable_time = variable           #Vary timestep with error (option for ease of debugging)
        self.orientation = np.array([launch_site.rail_yaw*np.pi/180,(launch_site.lat+launch_site.rail_pitch)*np.pi/180,0]) #yaw pitch roll  of the body frame in the inertial frame rad
        self.w = np.matmul(rot_matrix(self.orientation),np.array([ang_vel_earth,0,0]))          #rate of change of yaw pitch roll rad/s - would this have an initial value? I don't think it should since after it is free of the rail (which should be negligable)
        self.pos = pos_launch_to_inertial(np.array([0,0,0]),launch_site,0)                      #Position in inertial coordinates [x,y,z] m
        self.v = vel_launch_to_inertial([0,0,0],launch_site,0)                                  #Velocity in intertial coordinates [x',y',z'] m/s
        self.alt = launch_site.alt                                                              #Altitude
        self.on_rail=True
    
    def body_to_inertial(self,vector):  #Convert a vector in x,y,z to X,Y,Z
        return np.matmul(rot_matrix(self.orientation), np.array(vector))
    
    def aero_forces(self, alt, velocity):
        '''
        Returns aerodynamic forces (in the body reference frame, [x_b, y_b, z_b]), and the distance of the centre of pressure (COP) from the front of the vehicle.
        You can use these forces, and the position they act on (the COP), to find the moments about the centre of mass.
        
        Note that:
         -This currently ignores the damping moment generated by the rocket is rotating about its long axis
         -I'm not sure if I used the right angles for calculating CN as the angles of attack vary
             (same for CA as angles of attack vary)
         -Not sure if I'm using the right density for converting between force coefficients and forces
        '''
        print("Running Rocket.aero_forces()")

        #Velocities and Mach number
        v_rel_wind = velocity - vel_launch_to_inertial(self.launch_site.wind,self.launch_site,self.time)
        v_a = np.linalg.norm(v_rel_wind)
        v_sound = np.interp(alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.sdat)
        mach = v_a/v_sound
        
        #Angles - use np.angle(a + jb) to replace np.arctan(b/a) because the latter gave divide by zero errors, if x=0
        alpha = np.angle(1j*v_rel_wind[2] + v_rel_wind[0])
        beta = np.angle(1j*v_rel_wind[1] + (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.angle( 1j*(v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5 + v_rel_wind[0])
        alpha_star = np.angle(1j*v_rel_wind[2] + (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )
        beta_star = np.angle(1j*v_rel_wind[1] + v_rel_wind[0])
            
        #Dynamic pressure at the current altitude and velocity - WARNING: Am I using the right density?
        q = 0.5*np.interp(alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.ddat)*(v_a**2)
        
        #Characteristic area
        S = self.aero.area
        
        #Drag/Force coefficients
        Cx = self.aero.CA(mach, abs(delta))[0]         #WARNING: Not sure if I'm using the right angles for these all
        Cz = self.aero.CN(mach, abs(alpha_star))[0]    #Or if this is the correct way to use CN
        Cy = self.aero.CN(mach, abs(beta))[0] 
        
        #Forces
        Fx = -np.sign(v_rel_wind[0])*Cx*q*S                         
        Fy = -np.sign(v_rel_wind[1])*Cy*q*S                         
        Fz = -np.sign(v_rel_wind[2])*Cz*q*S
        
        #Position where moments act:
        COP = self.aero.COP(mach, abs(delta))[0]
        
        #Return the forces (note that they're given using the body coordinate system, [x_b, y_b, z_b]).
        #Also return the distance that the COP is from the front of the rocket.
        print("Finished running Rocket.aero_forces()")
        return np.array([Fx,Fy,Fz]), COP
        
    def thrust(self, time, alt, vector = [-1,0,0]): 
        '''
        Returns thrust and moments generated by the motor, in body-fixed coordinates [x_b, y_b, z_b]
        time - Time since motor ignition (s)
        vector - Direction of thrust (will automatically be converted to a unit vector), as a vector in the body coordinate system.
                    e.g. [-1,0,0] means the thrust points downwards (perfect alignment between thrust and the rocket body)
                    Used to model thrust misalignment or thrust vectoring.
        
        Notes:
            -This is almost all copied from Joe Hunt's original trajectory_sim.py
            -Thrust misalignment is not yet modelled, so this currently only outputs a thrust
            -It currently takes a 'time' argument, which is the time since motor ignition. We could change this to self.time later if we want.
        '''   
        print("Running Rocket.thrust()")

        #Make sure "vector" is a Numpy array, in case the user inputted a Python list.
        vector = np.array(vector)
        
        if time < max(self.motor.motor_time_data):
            #Get the motor parameters at the current moment in time
            pres_cham = np.interp(time, self.motor.motor_time_data, self.motor.cham_pres_data)
            dia_throat = np.interp(time, self.motor.motor_time_data, self.motor.throat_data)
            gamma = np.interp(time, self.motor.motor_time_data, self.motor.gamma_data)
            nozzle_efficiency = np.interp(time, self.motor.motor_time_data, self.motor.nozzle_efficiency_data)
            pres_exit = np.interp(time, self.motor.motor_time_data, self.motor.exit_pres_data)
            nozzle_area_ratio = np.interp(time, self.motor.motor_time_data, self.motor.area_ratio_data)
            
            #Get atmospheric pressure (to calculate pressure thrust)
            pres_static = np.interp(alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.padat)
            
            #Calculate the thrust
            area_throat = ((dia_throat/2)**2)*np.pi
            thrust = (area_throat*pres_cham*(((2*gamma**2/(gamma-1))
                                             *((2/(gamma+1))**((gamma+1)/(gamma-1)))
                                             *(1-(pres_exit/pres_cham)**((gamma-1)/gamma)))**0.5)
                     +(pres_exit-pres_static)*area_throat*nozzle_area_ratio)
    
            thrust *= nozzle_efficiency
        else:
            #If the engine has finished burning, no thrust is produced
            thrust = 0
        
        #Multiply the thrust by the direction it acts in, and return it.
        print("Finished running Rocket.thrust()")
        return thrust*vector/np.linalg.norm(vector)
        
    def gravity(self, time, position):
        '''
        Returns the gravity force, as a vector in inertial coordinates
        
        Uses a spherical Earth gravity model
        '''
        # F = -GMm/r^2 = μm/r^2 where μ = 3.986004418e14 for Earth
        return -3.986004418e14 * self.mass_model.mass(time) * position / np.linalg.norm(position)**3

    
    def altitude(self, position):
        return np.linalg.norm(position)-r_earth
    
    def acceleration(self, position, velocity, angular_velocity, time):     #Returns translational and rotational accelerations on the rocket, given the applied forces
        '''
        Returns lin_acc, rot_acc
        
        lin_acc = The linear acceleration in inertial coordinates [ax_i, ay_i, az_i]
        rot_acc = The rotataional acceleration in inertial coordinates [wdot_x_i, wdot_y_i, wdot_z_i]
        '''

        print("Running Rocket.acceleration()")
        
        #Get all the forces in body coordinates
        thrust_b = self.thrust(time,self.altitude(position))
        aero_force_b, cop = self.aero_forces(self.altitude(position),velocity)
        cog = self.mass_model.cog(time)
        
        
        #Get the vectors we need to calculate moments
        r_engine_cog_b = (self.mass_model.l - cog)*np.array([-1,0,0])   #Vector (in body coordinates) of nozzle exit, relative to CoG
        r_cop_cog_b = (cop - cog)*np.array([-1,0,0])                    #Vector (in body coordinates) of CoP, relative to CoG
        
        #Calculate moments in body coordinates using moment = r x F    
        aero_moment_b = np.cross(r_cop_cog_b, aero_force_b)
        thrust_moment_b = np.cross(r_engine_cog_b, thrust_b)
        
        #Convert forces to inertial coordinates
        thrust = self.body_to_inertial(thrust_b)
        aero_force = self.body_to_inertial(aero_force_b)
        
        #Get total force and moment
        F = thrust + aero_force + self.gravity(time, position)
        Q_b = aero_moment_b + thrust_moment_b   #Keep the moments in the body coordinate system for now
        
        #F = ma in inertial coordinates
        lin_acc = F/self.mass_model.mass(time)
        
        #Q = Ig wdot in body coordinates (because then we're using principal moments of inertia)
        rot_acc_b = np.array([Q_b[0]/self.mass_model.ixx(time),
                           Q_b[1]/self.mass_model.iyy(time),
                           Q_b[2]/self.mass_model.izz(time)])
        
        #Convert rotational accelerations into inertial coordinates
        rot_acc = self.body_to_inertial(rot_acc_b)
        
        print("Finished running Rocket.acceleration()")
        return lin_acc, rot_acc
    
    
    def step(self):
        """Implimenting a time step variable method based on Numerical recipes (link in readme)
        """
        k_1=self.h*self.acceleration(self.pos,self.v,self.w,self.time)
        k_2=self.h*self.acceleration(self.pos,self.v+a[1][0]*k_1,self.w,self.time+c[1]*self.h)
        k_3=self.h*self.acceleration(self.pos,self.v+a[2][0]*k_1+a[2][1]*k_2,self.w,self.time+c[2]*self.h)
        k_4=self.h*self.acceleration(self.pos,self.v+a[3][0]*k_1+a[3][1]*k_2+a[3][2]*k_3,self.w,self.time+c[3]*self.h)
        k_5=self.h*self.acceleration(self.pos,self.v+a[4][0]*k_1+a[4][1]*k_2+a[4][2]*k_3+a[4][3]*k_4,self.w,self.time+c[4]*self.h)
        k_6=self.h*self.acceleration(self.pos,self.v+a[5][0]*k_1+a[5][1]*k_2+a[5][2]*k_3+a[5][3]*k_4+a[5][4]*k_5,self.w,self.time+c[4]*self.h)
        
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
            scale_r=atol_r+rtol_r*abs(np.maximum.reduce([r,np.stack((self.pos,self.orientation))]))
            err_v=(v-v_)/scale_v
            err_r=(r-r_)/scale_r
            err=max(err_v,err_r)
            self.h=sf*self.h*pow(max(err),-1/5)

        self.v=v[0]
        self.w=v[1]
        self.pos=r[0]
        self.orientation=r[1]
        
def pos_launch_to_inertial(position,launch_site,time):
    """Converts position in launch frame to position in inertial frame
    Adapted from https://gist.github.com/mpkuse/d7e2f29952b507921e3da2d7a26d1d93 

    Args:
        position (Numpy Array): Position in launch frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition

    Returns:
        Numpy array: Velocity in inertial frame
    """
    phi = launch_site.lat / 180. * np.pi
    lambada = (launch_site.longi) / 180. * np.pi
    h = launch_site.alt

    position_rotated = np.matmul(rot_matrix(np.array([0,np.pi/2-phi,lambada]),True),position)
    e=e_earth
    q = np.sin( phi )
    N = r_earth / np.sqrt( 1 - e*e * q*q )
    X = (N + h) * np.cos( phi ) * np.cos( lambada )+position_rotated[0]
    Y = (N + h) * np.cos( phi ) * np.sin( lambada )+position_rotated[1]
    Z = (N*(1-e*e) + h) * np.sin( phi )-position_rotated[2]
    return np.matmul(rot_matrix(np.array([time*ang_vel_earth,0,0])),np.array([X,Y,Z]))

def pos_inertial_to_launch(position,launch_site,time):
    """Converts position in inertial frame to position in launch frame
    Adapted from https://gist.github.com/mpkuse/d7e2f29952b507921e3da2d7a26d1d93 

    Args:
        position (Numpy Array): Position in inertial frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition

    Returns:
        Numpy array: Velocity in launch frame
    """
    phi = launch_site.lat / 180. * np.pi
    lambada = (launch_site.longi) / 180. * np.pi+time*ang_vel_earth
    h = launch_site.alt

    e=e_earth
    q = np.sin( phi )
    N = r_earth / np.sqrt( 1 - e*e * q*q )
    X = position[0]-(N + h) * np.cos( phi ) * np.cos( lambada )
    Y = position[1]-(N + h) * np.cos( phi ) * np.sin( lambada )
    Z = position[2]-(N*(1-e*e) + h) * np.sin( phi )
    return np.matmul(rot_matrix(np.array([time*ang_vel_earth,np.pi/2-phi,lambada-time*ang_vel_earth]),True),np.array([X,Y,Z]))

def vel_inertial_to_launch(velocity,launch_site,time):
    """Converts inertial velocity to velocity in launch frame

    Args:
        velocity (np.array): [x,y,z] Velocity in inertial frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition

    Returns:
        Numpy array: Velocity in launch frame
    """    
    launch_site_velocity = np.array([0,ang_vel_earth*(r_earth+launch_site.alt)*np.cos(launch_site.lat*np.pi/180),0])
    inertial_rot_launch = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,0,0],True),velocity)
    return inertial_rot_launch-launch_site_velocity

def vel_launch_to_inertial(velocity,launch_site,time):
    """Converts launch frame velocity to velocity in inertial frame

    Args:
        velocity (Numpy array): [x,y,z] velocity in launch frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition

    Returns:
        Numpy array: Velocity in inertial frame
    """    
    launch_site_velocity = np.array([0,ang_vel_earth*(r_earth+launch_site.alt)*np.cos(launch_site.lat*np.pi/180),0])
    launch_rot_inertial = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,0,0]),velocity)
    return launch_rot_inertial+launch_site_velocity

def rot_matrix(orientation,inverse=False):
    """Generates a rotation matrix between frames which are rotated by yaw, pitch and roll specified by orientation
    Left hand matrix multiply this by the relivant vector (i.e. np.matmul(rotation_matrix(....),vec))

    Args:
        orientation (np.array): The rotation of the frames relative to eachother as yaw, pitch and roll (about z, about y, about x)
        inverse (bool, optional): If the inverse is required (i.e. when transforming from the frame which is rotated by orientation). Defaults to False.

    Returns:
        [type]: [description]
    """    
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

def run_simulation(rocket):
    """Runs the simulaiton to completeion outputting dictionary of the position, velocity and mass of the rocket

    Args:
        rocket (Rocket): The rocket to be simulated

    Returns:
        Pandas Dataframe: Record of position, velocity and mass at time t 
    """  
    record=pd.DataFrame({"Time":[],"Position":[],"Velocity":[],"Mass":[]}) #time:[position,velocity,mass]
    while rocket.alt>0:
        rocket.step()
        launch_position = pos_inertial_to_launch(rocket.pos,rocket.launch_site,rocket.time)
        launch_velocity = vel_inertial_to_launch(rocket.vel,rocket.launch_site,rocket.time)
        record.append({"Time":rocket.time,
                        "x":launch_position[0],
                        "y":launch_position[1],
                        "z":launch_position[2],
                        "v_x":launch_velocity[0],
                        "v_y":launch_velocity[1],
                        "v_z":launch_velocity[2],
                        "Mass":rocket.m}, ignore_index=True)
    return record

def get_velocity_magnitude(df):
    return (np.sqrt(df["v_x"]**2+df["v_y"]**2+df["v_z"]**2))

def plot_altitude_time(simulation_output):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(simulation_output["y"], -simulation_output["x"])
    axs[0, 0].set_title('Ground Track ($^*$)')
    axs[0,0].set_xlabel("East/m")
    axs[0,0].set_ylabel("North/m")
    #plt.text(0,-simulation_output["x"].max(),'$^*$ This is in the fixed cartesian launch frame so will not be actual ground position over large distances',horizontalalignment='left', verticalalignment='center')
    axs[0, 1].plot(simulation_output["Time"],simulation_output["z"], 'tab:orange')
    axs[0, 1].set_title('Altitude')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Altitude/m")
    axs[1, 0].plot(simulation_output["Time"],simulation_output.apply(get_velocity_magnitude,axis=1), 'tab:green')
    axs[1, 0].set_title('Speed')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Speed/m/s")
    axs[1, 1].plot(simulation_output["Time"],simulation_output["v_z"], 'tab:red')
    axs[1, 1].set_title('Vertical Velocity')
    axs[1,1].set_xlabel("Time/s")
    axs[1,1].set_ylabel("Velocity/m/s")
    fig.tight_layout()
    
    plt.show() 
