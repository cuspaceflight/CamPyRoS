"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
Nomenclature is the mostly same as used in https://apps.dtic.mil/sti/pdfs/AD0642855.pdf

COORDINATE SYSTEM NOMENCLATURE
x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
x_i,y_i,z_i = Inertial coordinate system (does not rotate, origin at centre of the Earth)
x_l, y_l, z_l = Launch site coordinate system (origin on launch site, rotates with the Earth)


Would be useful to define directions, e.g. maybe
- Body:
    y points east and z north at take off (before rail alignment is accounted for) x up

- Launch site:
    z points perpendicular to the earth, y in the east direction and x tangential to the earth pointing south
    
- Inertial:
    Origin at centre of the Earth
    Z points to north from centre of earth, x aligned with launchsite at start and y orthogonal

'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class Atmosphere:
    """Object holding atmospheric data
    """    
    def __init__(self, adat, ddat, sdat, padat): 
        """[summary]

        Args:
            adat (list): Altitude m
            ddat (list): Density kg/m^3
            sdat (list): Speed of Sound m/s
            padat (list): Pressure Pa
        """        
        self.adat = adat
        self.ddat = ddat
        self.sdat = sdat
        self.padat = padat

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
atol_v=np.array([[0.01,0.01,0.01],[0.0001,0.0001,0.0001]]) #absolute error of each component of v and w
rtol_v=np.array([[0.001,0.001,0.001],[0.00001,0.00001,0.00001]]) #relative error of each component of v and w
atol_r=np.array([[0.1,0.1,0.1],[0.01,0.01,0.01]]) #absolute error of each component of position and pointing
rtol_r=np.array([[0.01,0.01,0.01],[0.001,0.001,0.001]]) #relative error of each component of position and pointing
sf=0.98 #Safety factor for h scaling

r_earth = 6378137 #(earth's semimarjor axis in meters)
#e_earth = 0.081819191 #earth ecentricity
e_earth = 0 #for simplicity of other calculations for now - if changed need to update the launchsite orientation and velocity transforms
ang_vel_earth=7.292115090e-5

class Motor:
    """Object holding the pefoemance data for the engine
    """    
    def __init__(self, motor_time_data, prop_mass_data, cham_pres_data, throat_data,
                 gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data):
        """Set up motor

        Args:
            motor_time_data (list): Time since ignition s
            prop_mass_data (list): Mass remaning kg
            cham_pres_data (list): Chamber Pressure Pa
            throat_data (list): Throat diameter m
            gamma_data (list): Nozzle inlet gamma (ratio of specific heats)
            nozzle_efficiency_data (list): Nozzle efficiency
            exit_pres_data (list): Exit pressure Pa
            area_ratio_data (list): Area ratio 
        """        
        self.motor_time_data = motor_time_data
        self.prop_mass_data = prop_mass_data
        self.cham_pres_data = cham_pres_data
        self.throat_data = throat_data
        self.gamma_data = gamma_data
        self.nozzle_efficiency_data = nozzle_efficiency_data
        self.exit_pres_data = exit_pres_data
        self.area_ratio_data = area_ratio_data

class LaunchSite:
    """Object holding launch site information
    """
    def __init__(self, rail_length, rail_yaw, rail_pitch,alt, longi, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
        """Sets up launch site

        Args:
            rail_length (float): Length of launch rail m
            rail_yaw (float): Angle of rotation about the z axis (north pointing) rad
            rail_pitch (float): Angle of rotation about "East" pointing y axis - in order to simplify calculations below this needs to be measured in the yaw then pitch order rad
            alt (float): Altitude m
            longi (float): Longditude degrees
            lat (float): Latitude degrees
            wind (list, optional): Wind vector at launch site. Defaults to [0,0,0]. Will increase completness/complexity at some point to include at least altitude variation.
            atmosphere (Atmosphere Object, optional): Atmosphere object. Defaults to StandardAtmosphere.
        """        
        self.rail_length = rail_length
        self.rail_yaw = rail_yaw 
        self.rail_pitch = rail_pitch
        self.alt = alt
        self.longi = longi
        self.lat = lat
        self.wind = np.array(wind)
        self.atmosphere = atmosphere
        
class CylindricalMassModel:
    """Simple cylindrical model of the rockets mass and moments of inertia.  Assumes the rocket is a solid cylinder, constant volume, which has a mass that reduces with time (i.e. the density of the cylinder reducs)
    """    
    def __init__(self, mass, time, l, r):
        """[summary]

        Args:
            mass (list): A list of masses, at each moment in time (after ignition) kg
            time (list): A list of times, corresponding to the masses in the 'mass' list s
            l (float): length of the cylinder m
            r (float): radius of the cylinder m
        """        
        self.mass_interp = scipy.interpolate.interp1d(time, mass)
        self.time = time
        self.l = l
        self.r = r

    def mass(self, time):
        """Returns the mass at some time after igition

        Args:
            time (float): Time since ignition

        Raises:
            ValueError: Raised if time is negative

        Returns:
            Float: The mass
        """        
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            return self.mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.mass_interp(time)
        else:
            return self.mass_interp(self.time[-1])

    def ixx(self, time):
        """Returns the xx moment of inertia at some time after igition

        Args:
            time (float): Time since ignition

        Raises:
            ValueError: Raised if time is negative

        Returns:
            Float: The moment of inertia
        """       
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            return (1/2)* self.r**2 * self.mass(self.time[0])
        elif time < self.time[-1]:
            return (1/2)* self.r**2 * self.mass(time)
        else:
            return (1/2)* self.r**2 * self.mass(self.time[-1])

      
    def iyy(self, time):
        """Returns the yy moment of inertia at some time after igition

        Args:
            time (float): Time since ignition

        Raises:
            ValueError: Raised if time is negative

        Returns:
            Float: The moment of inertia 
        """    
        if time < 0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[0])
        elif time < self.time[-1]:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(time)
        else:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[-1])
      
    def izz(self, time):
        """Returns the zz moment of inertia at some time after igition

        Args:
            time (float): Time since ignition

        Raises:
            ValueError: Raised if time is negative

        Returns:
            Float: The moment of inertia
        """    
        return self.iyy(time)
    
    def cog(self, time):
        """Returns the center of gravity, as a distance from the tip of the nose.

        Args:
            time (float): Time since ignition

        Returns:
            float: The centre of gravity m
        """        
        return self.l/2

class RasAeroData:
    """Object holding aerodynamic data from a RasAero II 'Aero Plots' export file
    """    
    def __init__(self, file_location_string, area = 0.0305128422): 
        """Loads the RasAero data

        Args:
            file_location_string (string): Route to data file
            area (float, optional): Area used to normalise coefficients. Defaults to 0.0305128422. m^2

        """
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
            
            
class Rocket:
    """Object that hold the rocket information and functions
    """    
    def __init__(self, mass_model, motor, aero, launch_site, h, variable=False):
        """Sets up the rocket object

        Args:
            mass_model (MassModel): Model of mass development of the rocket, only current option is CylindricalMassModel
            motor (Motor Object): Motot objet
            aero (Aeroo Object): Object holding the aerodynamic data 
            launch_site (Launchsite Object): Launch site object
            h (float): Time step length
            variable (bool, optional): Adaptive timestep?. Defaults to False.
        """        

        self.launch_site = launch_site                  #LaunchSite object
        self.motor = motor                              #Motor object containing motor data
        self.aero = aero                                #object containing aerodynamic information
        self.mass_model = mass_model                    #Object used to model the mass of the rocket
        
        self.time = 0                           #Time since ignition (seconds)
        self.h = h                              #Time step size (can evolve)
        self.variable_time = variable           #Vary timestep with error (option for ease of debugging)
        self.orientation = np.array([launch_site.longi*np.pi/180,-launch_site.lat*np.pi/180,0])     #np.array([launch_site.rail_yaw*np.pi/180,(launch_site.rail_pitch)*np.pi/180+np.pi/2-launch_site.lat*np.pi/180,0]) #yaw pitch roll  of the body frame in the inertial frame rad
        self.w_i = np.matmul(rot_matrix(self.orientation),np.array([ang_vel_earth,0,0]))            #Angular velocity in inertial coordinates
        self.pos_i = pos_launch_to_inertial(np.array([0,0,0]),launch_site,0)                        #Position in inertial coordinates
        self.v_i = vel_launch_to_inertial([0,0,0],launch_site.lat, launch_site.longi,0,0)           #Velocity in intertial coordinates
        self.alt = launch_site.alt                                                                  #Altitude
        self.on_rail=True
    
    def body_to_inertial(self,vector,orientation):
        """Converts a vector in the body frame to the inertial frame

        Args:
            vector (Numpy Array): Vector to be transformed
            orientation (Numpy Array): Orientation of the body frame rad

        Returns:
            Numpy Array: Vector transformed to the inertial frame
        """        
        return np.matmul(rot_matrix(orientation), np.array(vector))

    def inertial_to_body(self,vector ,orientation):
        """Converts a vector in the inertial frame to the body frame

        Args:
            vector (Numpy Array): Vector to be transformed
            orientation (Numpy Array): Orientation of the body frame rad

        Returns:
            Numpy Array: Vector transformed to the body frame
        """    
        return np.matmul(rot_matrix(orientation, inverse=True), np.array(vector))

    def aero_forces(self, alt, orientation, position, velocity, time):
        """Returns aerodynamic forces (in the body reference frame and the distance of the centre of pressure (COP) from the front of the vehicle.
         Note that:
         -This currently ignores the damping moment generated by the rocket is rotating about its long axis
         -Unsure if the right angles for calculating CN as the angles of attack vary
             (same for CA as angles of attack vary)
         -Not sure if using the right density for converting between force coefficients and forces

        Args:
            alt (float): Curent Altitude of the rocket - used to recover atmospheric information
            orientation (Numpy Array): Orientation of the rocket in the inertial frame rad
            position (Numpy Array): Position of the rocket in the inertial frame m
            velocity (Numpy Array): Velocity of the rocket in the inertial frame m/s
            time (float): Time since ignition s

        Returns:
            Numpy Array: Returns force in the x,y,z directions in the body frame N
            Float: Centre of pressure m
        """             

        wind_inertial = vel_launch_to_inertial(self.launch_site.wind,np.arctan(position[2]/np.sqrt(position[0]**2+position[1]**2))*180/np.pi,np.arctan(round(position[0],10)/round(position[1],10))*180/np.pi,time,self.altitude(position))
        v_rel_wind = self.inertial_to_body(velocity-wind_inertial, orientation)
        v_a = np.linalg.norm(v_rel_wind)
        v_sound = np.interp(alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.sdat)
        mach = v_a/v_sound
        
        #Angles - use np.angle(a + jb) to replace np.arctan(a/b) because the latter gave divide by zero errors, if b=0
        alpha = np.angle(1j*v_rel_wind[2] + v_rel_wind[0])
        beta = np.angle(1j*v_rel_wind[1] + (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.angle( 1j*(v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5 + v_rel_wind[0])
        alpha_star = np.angle(1j*v_rel_wind[2] + (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )
        beta_star = np.angle(1j*v_rel_wind[1] + v_rel_wind[0])
        
        #If the angle of attack is too high, our linearised model will be inaccurate
        """if delta>2*np.pi*6/360:
            print("WARNING: delta = {:.2f} (Large angle of attack)".format(360*delta/(2*np.pi)))
        if alpha_star>2*np.pi*6/360:
            print("WARNING: alpha* = {:.2f} (Large angle of attack)".format(360*alpha_star/(2*np.pi)))
        if beta>2*np.pi*6/360:
            print("WARNING: beta = {:.2f} (Large angle of attack)".format(360*beta/(2*np.pi)))
        """
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
        return np.array([Fx,Fy,Fz]), COP
        
    def thrust(self, time, alt, vector = [1,0,0]): 
        """  Returns thrust and moments generated by the motor, in body frame. Mainly derived from Joe Hunt's original trajectory_sim.py

        Args:
            time (float): Time since ignition s
            alt (float): Altitude, used to recover atmospheric information
            vector (list, optional): Thrust direction - models miss alignment or thrust vector control. Defaults to [1,0,0].

        Returns:
            Numpy Array: Force on rocket x,y,z N
        """        
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
        return thrust*vector/np.linalg.norm(vector)
        
    def gravity(self, time, position):
        """ Returns the gravity force, as a vector in inertial coordinates.

        Args:
            time (float): Time since ignition, used to recover mass
            position (Numpy Array): Position in the inertial frame

        Returns:
            Numpy Array: Force on the body in the inertial frame
        """        
        '''
        Returns the gravity force, as a vector in inertial coordinates
        
        Uses a spherical Earth gravity model
        '''
        # F = -GMm/r^2 = μm/r^2 where μ = 3.986004418e14 for Earth
        return -3.986004418e14 * self.mass_model.mass(time) * position / np.linalg.norm(position)**3

    
    def altitude(self, position):
        """Returns the altitude corresponding to a position in the inertial frame, can be complicated in the future to account for non spherical nature of Earth

        Args:
            position (Numpy Array): Position in the inertial frame m

        Returns:
            Float: Altitude m
        """        
        return np.linalg.norm(position)-r_earth
    
    def acceleration(self, positions, velocities, time):
        """Returns translational and rotational accelerations on the rocket, given the applied forces

        Args:
            positions (Numpy Arrray): The translational and rotational positions in the inertial frame (position stacked with orientation)
            velocities (Numpy Arrray): The translational and rotational velocities in the inertial frame
            time (Float): Time since igition

        Returns:
            [Numpy Array]: Translational accleration in inertial frame, and rotational acceleration using the body coordinate system
        """        
        
        #Get all the forces in body coordinates
        thrust_b = self.thrust(time,self.altitude(positions[0]))
        aero_force_b, cop = self.aero_forces(self.altitude(positions[0]),positions[1],positions[0],velocities[0],time)
        cog = self.mass_model.cog(time)
    
        #Get the moment arms
        r_engine_cog_b = (self.mass_model.l - cog)*np.array([-1,0,0])   #Vector (in body coordinates) of nozzle exit, relative to CoG
        r_cop_cog_b = (cop - cog)*np.array([-1,0,0])                    #Vector (in body coordinates) of CoP, relative to CoG
        
        #Calculate moments in body coordinates using moment = r x F    
        aero_moment_b = np.cross(r_cop_cog_b, aero_force_b)
        thrust_moment_b = np.cross(r_engine_cog_b, thrust_b)
        
        #Convert forces to inertial coordinates
        thrust = self.body_to_inertial(thrust_b,positions[1])
        aero_force = self.body_to_inertial(aero_force_b,positions[1])
        
        #Get total force and moment
        F = thrust + aero_force + self.gravity(time, positions[0])
        Q_b = aero_moment_b + thrust_moment_b   
        
        #Calculate angular velocities using Euler's equations - IIA Engineering, Module 3C5, Rigid body dynamics handout (page 18)
        i_b = np.array([self.mass_model.ixx(time),
                        self.mass_model.iyy(time),
                        self.mass_model.izz(time)])     #Moments of inertia [ixx, iyy, izz]
        w_b=np.matmul(rot_matrix(positions[1]),velocities[1])
        wdot_b = np.array([(Q_b[0] + (i_b[1] - i_b[2])*w_b[1]*w_b[2]) / i_b[0]
                            ,(Q_b[1] + (i_b[2] - i_b[0])*w_b[2]*w_b[0]) / i_b[1]
                            ,(Q_b[2] + (i_b[0] - i_b[1])*w_b[0]*w_b[1]) / i_b[2]])                      #Initialise empty array
        if self.on_rail==True:
            F=np.array([F[0],0,0])
            wdot_b=np.array([wdot_b[0],0,0])
        
        #F = ma in inertial coordinates
        lin_acc = F/self.mass_model.mass(time)

        wdot_i=np.matmul(rot_matrix(positions[1]).transpose(),wdot_b)#https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
        return np.stack([lin_acc, wdot_i])
    
    def step(self):
        """Implimenting a time step variable method based on Numerical recipes (link in readme)
        """
        vels=np.stack([self.v_i,self.w_i])
        positions=np.stack([self.pos_i,self.orientation])
        k_1=self.h*self.acceleration(positions,vels,self.time)
        l_1=self.h*vels
        k_2=self.h*self.acceleration(positions+a[1][0]*l_1,vels+a[1][0]*k_1,self.time+c[1]*self.h)
        l_2=l_1+a[1][0]*k_1
        k_3=self.h*self.acceleration(positions+a[2][0]*l_1+a[2][1]*l_2,vels+a[2][0]*k_1+a[2][1]*k_2,self.time+c[2]*self.h)
        l_3=l_1+a[2][0]*k_1+a[2][1]*k_2
        k_4=self.h*self.acceleration(positions+a[3][0]*l_1+a[3][1]*l_2+a[3][2]*l_3,vels+a[3][0]*k_1+a[3][1]*k_2+a[3][2]*k_3,self.time+c[3]*self.h)
        l_4=l_1+a[3][0]*k_1+a[3][1]*k_2+a[3][2]*k_3
        k_5=self.h*self.acceleration(positions+a[4][0]*l_1+a[4][1]*l_2+a[4][2]*l_3+a[4][3]*l_4,vels+a[4][0]*k_1+a[4][1]*k_2+a[4][2]*k_3+a[4][3]*k_4,self.time+c[4]*self.h)
        l_5=l_1+a[4][0]*k_1+a[4][1]*k_2+a[4][2]*k_3+a[4][3]*k_4
        k_6=self.h*self.acceleration(positions+a[5][0]*l_1+a[5][1]*l_2+a[5][2]*l_3+a[5][3]*l_4+a[5][4]*l_5,vels+a[5][0]*k_1+a[5][1]*k_2+a[5][2]*k_3+a[5][3]*k_4+a[5][4]*k_5,self.time+c[4]*self.h)
        l_6=l_1+a[5][0]*k_1+a[5][1]*k_2+a[5][2]*k_3+a[5][3]*k_4+a[5][4]*k_5

        v=np.stack((self.v_i,self.w_i))+b[0]*k_1+b[1]*k_2+b[2]*k_3+b[3]*k_4+b[4]*k_5+b[5]*k_6 #+O(h^6)
        r=np.stack((self.pos_i,self.orientation))+(b[0]*l_1+b[1]*l_2+b[2]*l_3+b[3]*l_4+b[4]*l_5+b[5]*l_6) #+O(h^6) This is movement of the body frame from wherever it is- needs to be transformed to inertial frame

        self.time+=self.h
        if(self.variable_time==True):
            v_=np.stack((self.v_i,self.w_i))+b_[0]*k_1+b_[1]*k_2+b_[2]*k_3+b_[3]*k_4+b_[4]*k_5+b_[5]*k_6 #+O(h^5)
            r_=np.stack((self.pos,self.orientation))+b_[0]*l_1+b_[1]*l_2+b_[2]*l_3+b_[3]*l_4+b_[4]*l_5+b_[5]*l_6 #+O(h^5)

            scale_v=atol_v+rtol_v*abs(np.maximum.reduce([v,np.stack((self.v_i,self.w))]))
            scale_r=atol_r+rtol_r*abs(np.maximum.reduce([r,np.stack((self.pos,self.orientation))]))
            err_v=(v-v_)/scale_v
            err_r=(r-r_)/scale_r
            err=max(np.max(err_v),np.max(err_r))
            self.h=sf*self.h*pow(err,-1/5)

        self.v_i=v[0]
        self.w_i=v[1]
        self.pos_i=r[0]
        self.orientation=r[1]
    
    def check_phase(self):
        if self.on_rail==True:
            flight_distance = np.linalg.norm(pos_inertial_to_launch(self.pos_i,self.launch_site,self.time))
            if flight_distance>=self.launch_site.rail_length:
                print("Cleared rail at t=%ss with alt=%sm and TtW=%sG"%(self.time,
                self.altitude(self.pos_i),
                np.linalg.norm(self.acceleration([self.pos_i,self.orientation], [self.v_i, self.w_i], self.time)[0])/9.81)
                )
                self.on_rail=False

        
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

def inertial_to_inertial_long_lat(position):
    """ ECEF --> lat (PHI), long (LAMBDA)
    algorithm2 https://hal.archives-ouvertes.fr/hal-01704943v2/document
    From https://gist.github.com/mpkuse/d7e2f29952b507921e3da2d7a26d1d93
    """

    a = r_earth
    e = e_earth
    b = a * np.sqrt( 1.0 - e*e )
    _X = ecef_X[0]
    _Y = ecef_X[1]
    _Z = ecef_X[2]

    w_2 = _X**2 + _Y**2
    l = e**2 / 2.0
    m = w_2 / a**2
    n = _Z**2 * (1.0 - e*e) / (a*a)
    p = (m+n - 4*l*l)/6.0
    G = m*n*l*l
    H = 2*p**3 + G

    C = np.cbrt( H+G+2*np.sqrt(H*G) ) / np.cbrt(2)
    i = -(2.*l*l + m + n ) / 2.0
    P = p*p
    beta = i/3.0 - C -P/C
    k = l*l * ( l*l - m - n )
    t = np.sqrt( np.sqrt( beta**2 - k ) - (beta+i)/2.0 ) - np.sign( m-n ) * np.sqrt( np.abs(beta-i) / 2.0 )
    F = t**4 + 2*i*t*t + 2.*l*(m-n)*t + k
    dF_dt = 4*t**3 + 4*i*t + 2*l*(m-n)
    delta_t = -F / dF_dt
    u = t + delta_t + l
    v = t + delta_t - l
    w = np.sqrt( w_2 )
    __phi = np.arctan2( _Z*u, w*v )
    delta_w = w* (1.0-1.0/u )
    delta_z = _Z*( 1- (1-e*e) / v )
    h = np.sign( u-1.0 ) * np.sqrt( delta_w**2 + delta_z**2 )
    __lambda = np.arctan2( _Y, _X )


    return (__phi, __lambda)

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
    inertial_rot_launch = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,launch_site.lat*np.pi/180+np.pi/2,0],True),velocity)
    return inertial_rot_launch-launch_site_velocity

def vel_launch_to_inertial(velocity,launch_site_lat,launch_site_longi,time,alt):
    """Converts launch frame velocity to velocity in inertial frame
    Args:
        velocity (Numpy array): [x,y,z] velocity in launch frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in inertial frame
    """    
    launch_site_velocity = np.array([0,ang_vel_earth*(r_earth+alt)*np.cos(launch_site_lat*np.pi/180),0])
    ##inputlaunch_site_lat)
    ##inputlaunch_site_velocity)
    launch_rot_inertial = np.matmul(rot_matrix([time*ang_vel_earth+launch_site_longi*np.pi/180,0,0]),velocity)
    ##inputlaunch_rot_inertial)
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
    c=0
    d=0
    """Runs the simulaiton to completeion outputting dictionary of the position, velocity and mass of the rocket

    Args:
        rocket (Rocket): The rocket to be simulated

    Returns:
        Pandas Dataframe: Record of position, velocity and mass at time t 
    """  
    print("Running simulation")
    record=pd.DataFrame({"Time":[],"x":[],"y":[],"z":[],"v_x":[],"v_y":[],"v_z":[]}) #time:[position,velocity,mass]
    while (rocket.altitude(rocket.pos_i)>=0 and rocket.time<200):
        rocket.check_phase()
        rocket.step()
        launch_position = pos_inertial_to_launch(rocket.pos_i,rocket.launch_site,rocket.time)
        launch_velocity = vel_inertial_to_launch(rocket.v_i,rocket.launch_site,rocket.time)
        aero_forces, cop = rocket.aero_forces(rocket.altitude(rocket.pos_i), rocket.orientation, rocket.pos_i,rocket.v_i, rocket.time)
        cog = rocket.mass_model.cog(rocket.time)
        orientation = rocket.orientation
        x_b_l = orientation+np.array([np.pi/2+rocket.launch_site.rail_yaw*np.pi/180+ang_vel_earth*rocket.time,rocket.launch_site.rail_pitch*np.pi/180+np.pi/2-rocket.launch_site.lat*np.pi/180,0])
        x_b_i = rocket.body_to_inertial([1,0,0],rocket.orientation)
        #x_b_i = vel_inertial_to_launch(x_b_i, rocket.launch_site, rocket.time)
        lin_acc, rot_acc = rocket.acceleration([rocket.pos_i,rocket.orientation], [rocket.v_i, rocket.w_i], rocket.time)
        burnout_time = rocket.motor.motor_time_data[-1]
        new_row={"Time":rocket.time,
                        "x_i":rocket.pos_i[0],
                        "y_i":rocket.pos_i[1],
                        "z_i":rocket.pos_i[2],
                        "x_l":launch_position[0],
                        "y_l":launch_position[1],
                        "z_l":launch_position[2],
                        "vx_l":launch_velocity[0],
                        "vy_l":launch_velocity[1],
                        "vz_l":launch_velocity[2],
                        "aero_xb":aero_forces[0],
                        "aero_yb":aero_forces[1],
                        "aero_zb":aero_forces[2],
                        "cop":cop,
                        "cog":cog,
                        "orientation_0":orientation[0],
                        "orientation_1":orientation[1],
                        "orientation_2":orientation[2],
                        "attitude_ix":x_b_i[0],
                        "attitude_iy":x_b_i[1],
                        "attitude_iz":x_b_i[2],
                        "rot_acc_xi":rot_acc[0],
                        "rot_acc_yi":rot_acc[1],
                        "rot_acc_zi":rot_acc[2],
                        "h":rocket.h,
                        "burnout_time":burnout_time}
        record=record.append(new_row, ignore_index=True)
        if d==1000:
            print("t={} s (with h={} s)".format(rocket.time,rocket.h))
            d=0
        #print("alt={:.0f} time={:.1f}".format(rocket.altitude(rocket.pos), rocket.time))

        c+=1
        d+=1
    return record

def get_velocity_magnitude(df):
    return (np.sqrt(df["vx_l"]**2+df["vy_l"]**2+df["vz_l"]**2))

def plot_altitude_time(simulation_output):

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(simulation_output["y_l"], -simulation_output["x_l"])
    axs[0, 0].set_title('Ground Track ($^*$)')
    axs[0,0].set_xlabel("East/m")
    axs[0,0].set_ylabel("North/m")
    #plt.text(0,-simulation_output["x"].max(),'$^*$ This is in the fixed cartesian launch frame so will not be actual ground position over large distances',horizontalalignment='left', verticalalignment='center')
    axs[0, 1].plot(simulation_output["Time"],simulation_output["z_l"], 'tab:orange')
    axs[0, 1].set_title('Altitude')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Altitude/m")
    axs[1, 0].plot(simulation_output["Time"],simulation_output.apply(get_velocity_magnitude,axis=1), 'tab:green')
    axs[1, 0].set_title('Speed')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Speed/m/s")
    axs[1, 1].plot(simulation_output["Time"],-simulation_output["vz_l"], 'tab:red')
    axs[1, 1].set_title('Vertical Velocity')
    axs[1,1].set_xlabel("Time/s")
    axs[1,1].set_ylabel("Velocity/m/s")
    fig.tight_layout()

    plt.show() 

def plot_aero_forces(simulation_output):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(simulation_output["Time"], simulation_output["aero_xb"])
    axs[0, 0].set_title('aero_x_b')
    axs[0,0].set_xlabel("Time/s")
    axs[0,0].set_ylabel("Force/N")
    
    axs[0, 1].plot(simulation_output["Time"], simulation_output["aero_yb"])
    axs[0, 1].set_title('aero_y_b')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Force/N")
    
    axs[1, 0].plot(simulation_output["Time"], simulation_output["aero_zb"])
    axs[1, 0].set_title('aero_z_b')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Force/N")
    
    axs[1, 1].plot(simulation_output["Time"], -simulation_output["cop"], label="CoP")
    axs[1, 1].plot(simulation_output["Time"], -simulation_output["cog"], label="CoG")
    axs[1, 1].set_title('Centre of Pressure (distance from nose tip)')
    axs[1,1].set_xlabel("Time/s")
    axs[1,1].set_ylabel("Distance/m")
    
    axs[1,1].legend()
    
    plt.show()
    
def plot_velocity(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["Time"], simulation_output["vx_l"])
    axs[0, 0].set_title('V_x')
    axs[0,0].set_xlabel("Time/s")
    axs[0,0].set_ylabel("Velocity/m/s")
    
    axs[0, 1].plot(simulation_output["Time"], simulation_output["vy_l"])
    axs[0, 1].set_title('V_y')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Velocity/m/s")
    
    axs[1, 0].plot(simulation_output["Time"], simulation_output["vz_l"])
    axs[1, 0].set_title('V_z')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Velocity/m/s")
    plt.show()

def plot_position(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["Time"], simulation_output["x_l"])
    axs[0, 0].set_title('x')
    axs[0,0].set_xlabel("Time/s")
    axs[0,0].set_ylabel("Distance/m")
    
    axs[0, 1].plot(simulation_output["Time"], simulation_output["y_l"])
    axs[0, 1].set_title('y')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Distance/m")
    
    axs[1, 0].plot(simulation_output["Time"], simulation_output["z_l"])
    axs[1, 0].set_title('z')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Distance/m")
    plt.show()

def plot_orientation(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["Time"], simulation_output["orientation_0"])
    axs[0, 0].set_title('Yaw')
    axs[0,0].set_xlabel("Time/s")
    axs[0,0].set_ylabel("Angles/ rad")
    
    axs[0, 1].plot(simulation_output["Time"], simulation_output["orientation_1"])
    axs[0, 1].set_title('Pitch')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Angles/ rad")
    
    axs[1, 0].plot(simulation_output["Time"], simulation_output["orientation_2"])
    axs[1, 0].set_title('Roll')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Angles/ rad")

    axs[1, 1].plot(simulation_output["Time"], simulation_output["orientation_0"])
    axs[1, 1].plot(simulation_output["Time"], simulation_output["orientation_1"])
    axs[1, 1].plot(simulation_output["Time"], simulation_output["orientation_2"])
    axs[1, 1].set_title('Rocket.orientation[2]')
    axs[1,1].set_xlabel("Time/s")
    axs[1,1].set_ylabel("Angles/ rad")
    axs[1,1].legend(["Z","Y","X"])
    
    plt.show()

def plot_rot_acc(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["Time"], simulation_output["rot_acc_zi"])
    axs[0, 0].set_title('rot_acc_xi')
    axs[0,0].set_xlabel("Time/s")
    axs[0,0].set_ylabel("Acceleration/ rad s^-2")
    
    axs[0, 1].plot(simulation_output["Time"], simulation_output["rot_acc_yi"])
    axs[0, 1].set_title('rot_acc_yi')
    axs[0,1].set_xlabel("Time/s")
    axs[0,1].set_ylabel("Acceleration/ rad s^-2")
    
    axs[1, 0].plot(simulation_output["Time"], simulation_output["rot_acc_xi"])
    axs[1, 0].set_title('rot_acc_zi')
    axs[1,0].set_xlabel("Time/s")
    axs[1,0].set_ylabel("Acceleration/ rad s^-2")
    
    plt.show()

def plot_trajectory_3d(simulation_output, show_orientation=False):
    '''
    Plots the trajectory in 3D, given the simulation_output
    '''
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    x=simulation_output["x_i"]
    y=simulation_output["y_i"]
    z=simulation_output["z_i"]
    
    #Plot rocket position and launch site position
    ax.plot3D(x,y,z)
    ax.scatter(x[0],y[0],z[0],c='red', label="Launch site")
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')  
    ax.legend()
    
    #Plot the direction the rocket faces at each point (i.e. direction of x_b), using quivers
    if show_orientation==True:
        u=simulation_output["attitude_ix"]
        v=simulation_output["attitude_iy"]
        w=simulation_output["attitude_iz"]
        
        #Spaced out arrows, so it's not cluttered
        idx = np.round(np.linspace(0, len(u) - 1, int(len(u)/30))).astype(int)
        
        ax.quiver(x[idx], y[idx], z[idx], u[idx], v[idx], w[idx], length=1000, normalize=True, color="red")
        
    #Make axes equal ratios - source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to 
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    plt.show() 
    
    
def animate(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    #Get data
    yaw=simulation_output["orientation_0"]
    pitch=simulation_output["orientation_1"]
    roll=simulation_output["orientation_2"]
    altitude = simulation_output["z_l"]
    time = simulation_output["Time"]
    burnout_time = simulation_output["burnout_time"]
    #Set number of animation frames in total - less means that the animations runs faster
    frames = 500
    
    #Add titles
    axs[0, 0].set_title('Yaw')
    axs[0, 1].set_title('Pitch')
    axs[1, 0].set_title('Roll')
    axs[1, 1].set_title('Altitude / m')
    axs[1, 1].set_xlabel('Time / s')
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()    
    
    #Plot the initial directions
    axs[0,0].plot(np.linspace(0,np.cos(yaw[0]), 100), np.linspace(0, np.sin(yaw[0]), 100), lw=3, color='black')
    axs[0,1].plot(np.linspace(0,np.cos(pitch[0]), 100), np.linspace(0, np.sin(pitch[0]), 100), lw=3, color='black')
    axs[1,0].plot(np.linspace(0,np.cos(roll[0]), 100), np.linspace(0, np.sin(roll[0]), 100), lw=3, color='black')

    #Plot the point of engine burnout on the altitude-time graph
    burnout_index = (np.abs(time - burnout_time)).argmin()
    axs[1,1].scatter(time[burnout_index], altitude[burnout_index], color="red", label="Engine burnout")
    axs[1,1].legend()
    
    #Set up the lines that will be animated
    line1, = axs[0,0].plot([], [], lw=3, color='red')
    line2, = axs[0,1].plot([], [], lw=3, color='green')
    line3, = axs[1,0].plot([], [], lw=3, color='blue')
    line4, = axs[1,1].plot([], [], lw=3, color='orange')
    
    axs[0,0].set_xlim(-1,1)
    axs[0,0].set_ylim(-1,1)
    
    axs[0,1].set_xlim(-1,1)
    axs[0,1].set_ylim(-1,1)
    
    axs[1,0].set_xlim(-1,1)
    axs[1,0].set_ylim(-1,1)
    
    axs[1,1].set_xlim(0,200)
    axs[1,1].set_ylim(0,30e3)
    
    def init1():
        line1.set_data([], [])
        return line1,
    
    def init2():
        line2.set_data([], [])
        return line1,
    
    def init3():
        line3.set_data([], [])
        return line1,
    
    def init4():
        line4.set_data([], [])
        return line1,
    
    def animate1(i):
        j = int(i*len(yaw)/frames)
        x = np.linspace(0,np.cos(yaw[j]), 100)
        y = np.linspace(0, np.sin(yaw[j]), 100)
        line1.set_data(x, y)
        return line1,
    
    def animate2(i):
        j = int(i*len(pitch)/frames)
        x = np.linspace(0,np.cos(pitch[j]), 100)
        y = np.linspace(0, np.sin(pitch[j]), 100)
        line2.set_data(x, y)
        return line2,
    
    def animate3(i):
        j = int(i*len(roll)/frames)
        x = np.linspace(0,np.cos(roll[j]), 100)
        y = np.linspace(0, np.sin(roll[j]), 100)
        line3.set_data(x, y)
        return line3,
    
    def animate4(i):
        j = int(i*len(altitude)/frames)
        y = altitude[0:j]
        x = time[0:j]
        line4.set_data(x, y)
        return line4,
    
    
    anim1 = FuncAnimation(fig, animate1, init_func=init1,
                                   frames=frames, interval=20, blit=True)
    anim2 = FuncAnimation(fig, animate2, init_func=init2,
                                   frames=frames, interval=20, blit=True)
    anim3 = FuncAnimation(fig, animate3, init_func=init3,
                                   frames=frames, interval=20, blit=True)
    anim4 = FuncAnimation(fig, animate4, init_func=init4,
                                   frames=frames, interval=20, blit=True)
    
    
    plt.show()