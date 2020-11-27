"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
COORDINATE SYSTEM NOMENCLATURE
x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
x_i,y_i,z_i = Inertial coordinate system (does not rotate, origin at centre of the Earth)
x_l, y_l, z_l = Launch site coordinate system - the origin is on launch site but dropped down to a position of zero altitude (whilst keeping the same long and lat). Rotates with the Earth.

Directions are defined below.
- Body:
    y points east and z north at take off (before rail alignment is accounted for) x up.
    x is along the "long" axis of the rocket.

- Launch site:
    z points perpendicular to the earth, y in the east direction and x tangential to the earth pointing south
    
- Inertial:
    Origin at centre of the Earth
    z points to north from centre of earth, x aligned with launchsite at start and y orthogonal

'''

import csv, warnings
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.integrate as integrate


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

r_earth = 6378137               #(Earth's semimajor axis in meters)
ang_vel_earth=7.292115090e-5    #rads / s

#e_earth = 0.081819191 #earth ecentricity
e_earth = 0 #for simplicity of other calculations for now - if changed need to update the launchsite orientation and velocity transforms


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
    def __init__(self, rail_length, rail_yaw, rail_pitch, alt, longi, lat, wind=[0,0,0], atmosphere=StandardAtmosphere):
        """Sets up launch site
        Args:
            rail_length (float): Length of launch rail m
            rail_yaw (float): Angle of rotation (using a right hand rule) about the zl axis (i.e. "up") degrees. rail_yaw = 0 points South, rail_yaw = 90 points East.
            rail_pitch (float): Angle between the rail and the zl axis (i.e. angle to the vertical) degrees. rail_pitch = 0 points up.
            alt (float): Altitude m
            long (float): Longditude degrees
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
    def __init__(self, mass_model, motor, aero, launch_site, h, variable=False, rtol=1e-7, atol=1e-14):
        """Sets up the rocket object

        Args:
            mass_model (MassModel): Model of mass development of the rocket, only current option is CylindricalMassModel
            motor (Motor Object): Motot objet
            aero (Aero Object): Object holding the aerodynamic data 
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
        self.atol = atol
        self.rtol = rtol
        
        #Get the additional bit due to the angling of the rail
        rail_rotation = Rotation.from_euler('yz', [self.launch_site.rail_pitch, self.launch_site.rail_yaw], degrees=True)
    
        #Initialise the rocket's orientation - store it in a scipy.spatial.transform.Rotation object 
        xb_l = rail_rotation.apply([0,0,1])
        yb_l = rail_rotation.apply([0,1,0])
        zb_l = rail_rotation.apply([-1,0,0])

        xb_i = direction_l2i(xb_l, self.launch_site, self.time)     #xb should point up, and zl points up
        yb_i = direction_l2i(yb_l, self.launch_site, self.time)     #y for the body is aligned with y for the launch site (both point East)
        zb_i = direction_l2i(zb_l, self.launch_site, self.time)     #xl points South, zb should point North

        mat_b2i = np.zeros([3,3])
        mat_b2i[:,0] = xb_i
        mat_b2i[:,1] = yb_i
        mat_b2i[:,2] = zb_i
        self.b2i = Rotation.from_matrix(mat_b2i)     

        self.b2i = self.b2i                     #Body-to-Inertial Rotation - you can apply it to a vector with self.b2i.apply(vector)
        self.i2b = self.b2i.inv()               #Inertial-to-Body Rotation

        #Initialise angular positions and angular velocities
        self.pos_i = pos_l2i(np.array([0, 0, launch_site.alt]), launch_site, 0)                      #Position in inertial coordinates - defining the launch site origin as at an altitude of zero
        self.vel_i = vel_l2i([0,0,0], launch_site, 0)                                                #Velocity in intertial coordinates

        self.w_b = np.array([0,0,0])                                                 #Angular velocity in body coordinates
        self.alt = launch_site.alt                                                   #Altitude
        self.on_rail=True
        self.burn_out=False

    def aero_forces(self, b2i, pos_i, velocity, time):
        """Returns aerodynamic forces (in the body reference frame and the distance of the centre of pressure (COP) from the front of the vehicle.)
         Note that:
         -This currently ignores the damping moment generated by the rocket is rotating about its long axis
         -Unsure if the right angles for calculating CN as the angles of attack vary
             (same for CA as angles of attack vary)
         -Not sure if using the right density for converting between force coefficients and forces
        """             
        #Use np.angle(ja + b) to replace np.arctan(a/b)
        alt = self.altitude(pos_i)
        wind_inertial = vel_l2i(self.launch_site.wind, self.launch_site, time)
        v_rel_wind = b2i.inv().apply(velocity-wind_inertial)
        v_a = np.linalg.norm(v_rel_wind)
        v_sound = np.interp(alt, self.launch_site.atmosphere.adat, self.launch_site.atmosphere.sdat)
        mach = v_a/v_sound
        
        #Angles - use np.angle(ja + b) to replace np.arctan(a/b) because the latter gave divide by zero errors, if b=0
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
            if self.burn_out==False:
                print("Burnout at t={:.2f} s ".format(time))
            self.burn_out=True
        
        #Multiply the thrust by the direction it acts in, and return it.
        return thrust*vector/np.linalg.norm(vector)
        
    def gravity(self, time, pos_i):
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
        return -3.986004418e14 * self.mass_model.mass(time) * pos_i / np.linalg.norm(pos_i)**3
    
    def altitude(self, pos_i):
        """Returns the altitude corresponding to a position in the inertial frame, can be complicated in the future to account for non spherical nature of Earth

        Args:
            position (Numpy Array): Position in the inertial frame m

        Returns:
            Float: Altitude m
        """        
        return np.linalg.norm(pos_i)-r_earth

    def accelerations(self, pos_i, vel_i, w_b, b2i, time):
        """Returns translational and rotational accelerations on the rocket, given the applied forces

        Args:
            pos_i (Numpy Arrray): Inertial position
            ypr (Numpy Array): Yaw-pitch-roll array
            v_i (Numpy Arrray): Inertial velocity
            w_b (Numpy Arrray): Angular velocity in body coordinates
            time (Float): Time since igition

        Returns:
            [Numpy Array]: Translational accleration in inertial frame, and rotational acceleration using the body coordinate system
        """        
        
        #Get all the forces in body coordinates
        thrust_b = self.thrust(time,self.altitude(pos_i))
        aero_force_b, cop = self.aero_forces(b2i, pos_i, vel_i, time)
        cog = self.mass_model.cog(time)

        #Get the moment arms
        r_engine_cog_b = (self.mass_model.l - cog)*np.array([-1,0,0])   #Vector (in body coordinates) of nozzle exit, relative to CoG
        r_cop_cog_b = (cop - cog)*np.array([-1,0,0])                    #Vector (in body coordinates) of CoP, relative to CoG
        
        #Calculate moments in body coordinates using (moment = r x F)    
        aero_moment_b = np.cross(r_cop_cog_b, aero_force_b)
        thrust_moment_b = np.cross(r_engine_cog_b, thrust_b)
        
        #Convert forces to inertial coordinates
        thrust_i = b2i.apply(thrust_b)
        aero_force_i = b2i.apply(aero_force_b)

        #Get total force and moment
        F = thrust_i + aero_force_i + self.gravity(time, pos_i)
        Q_b = aero_moment_b + thrust_moment_b   

        #Calculate angular velocities using Euler's equations - IIA Engineering, Module 3C5, Rigid body dynamics handout (page 18)
        i_b = np.array([self.mass_model.ixx(time),
                        self.mass_model.iyy(time),
                        self.mass_model.izz(time)])     #Moments of inertia [ixx, iyy, izz]

        wdot_b = np.array([(Q_b[0] + (i_b[1] - i_b[2])*w_b[1]*w_b[2]) / i_b[0]
                            ,(Q_b[1] + (i_b[2] - i_b[0])*w_b[2]*w_b[0]) / i_b[1]
                            ,(Q_b[2] + (i_b[0] - i_b[1])*w_b[0]*w_b[1]) / i_b[2]])

        #F = ma in inertial coordinates
        lin_acc = F/self.mass_model.mass(time)     

        #If on the rail:
        if self.on_rail==True:
            xb_i = b2i.apply([1,0,0])
            xb_i = xb_i/np.linalg.norm(xb_i)            #Normalise it just in case (but this step should be unnecessary)
            lin_acc = np.dot(lin_acc, xb_i)*xb_i        #Make it so we only keep the acceleration along the body's x-direction (i.e. in the forwards direction)
            wdot_b = np.array([0,0,0])                  #Assume no rotational acceleration on the rail

        #For debugging:
        #vel_l = vel_i2l(self.vel_i, self.launch_site, self.time)
        #print("vel_l angle to zl = {}".format(180/np.pi * np.arccos(np.dot([0,0,1], vel_l / np.linalg.norm(vel_l)) )))

        #pos_l = pos_i2l(self.pos_i, self.launch_site, self.time)
        #print("pos_l angle to zl = {}".format(180/np.pi * np.arccos(np.dot([0,0,1], pos_l / np.linalg.norm(pos_l)) )))

        #lin_acc_l = direction_i2l(lin_acc, self.launch_site, self.time)
        #print("lin_acc angle to zl = {}".format(180/np.pi * np.arccos(np.dot([0,0,1], lin_acc_l / np.linalg.norm(lin_acc_l)) )))

        return np.stack([lin_acc, wdot_b])

    def fdot(self, time, fn):
        '''
        f contains everything needed to full define the rocket
        'fdot' here is the same as 'ydot' in the 2P1 (2nd Year) Engineering Lagrangian dynamics notes RK4 section

        f = [pos_i, vel_i, w_b, xb, yb, zb]
        fdot = [vel_i, acc_i, w_bdot, xbdot, ybdot, zbdot]

        This returns fdot
        '''

        pos_i = np.array([fn[0],fn[1],fn[2]])
        vel_i = np.array([fn[3],fn[4],fn[5]])
        w_b = np.array([fn[6],fn[7],fn[8]])
        xb = np.array([fn[9],fn[10],fn[11]])
        yb = np.array([fn[12],fn[13],fn[14]])
        zb = np.array([fn[15],fn[16],fn[17]])
        
        b2imat = np.zeros([3,3])
        b2imat[:,0] = xb
        b2imat[:,1] = yb
        b2imat[:,2] = zb
        b2i = Rotation.from_matrix(b2imat)
        w_i = b2i.apply(w_b)

        #vel_i = vel_i
        acc_i, w_bdot = self.accelerations(pos_i, vel_i, w_b, b2i, time)

        #If a vector 'r' is rotating in the inertial frame, dr/dt = w_i x r
        xbdot = np.cross(w_i, xb)
        ybdot = np.cross(w_i, yb)
        zbdot = np.cross(w_i, zb)

        return np.array([vel_i[0],vel_i[1],vel_i[2], acc_i[0],acc_i[1],acc_i[2], w_bdot[0],w_bdot[1],w_bdot[2], xbdot[0],xbdot[1],xbdot[2], ybdot[0],ybdot[1],ybdot[2], zbdot[0],zbdot[1],zbdot[2]])

    def run(self,max_time=300,verbose_log=False,debug=False,):
        d,c=0,0
        print("Running simulation")

        xb = self.b2i.as_matrix()[:,0]
        yb = self.b2i.as_matrix()[:,1]
        zb = self.b2i.as_matrix()[:,2]

        fn = [self.pos_i[0],self.pos_i[1],self.pos_i[2],self.vel_i[0],self.vel_i[1],self.vel_i[2], self.w_b[0],self.w_b[1],self.w_b[2], xb[0],xb[1],xb[2],yb[0],yb[1],yb[2], zb[0],zb[1],zb[2]]
        integrator = integrate.DOP853(self.fdot,0,fn,1000,atol=self.atol,rtol=self.rtol)

        record=pd.DataFrame({})

        while (self.altitude(self.pos_i)>=0 and self.time<max_time):
            if self.variable_time==False:
                integrator.h_abs=self.h
            events=self.check_phase()
            integrator.step()
            self.pos_i = np.array([integrator.y[0],integrator.y[1],integrator.y[2]])
            self.vel_i = np.array([integrator.y[3],integrator.y[4],integrator.y[5]])
            self.w_b = np.array([integrator.y[6],integrator.y[7],integrator.y[8]])
            b2imat = np.zeros([3,3])
            b2imat[:,0] = np.array([integrator.y[9],integrator.y[10],integrator.y[11]])   #body x-direction
            b2imat[:,1] = np.array([integrator.y[12],integrator.y[13],integrator.y[14]])      #body y-direction
            b2imat[:,2] = np.array([integrator.y[15],integrator.y[16],integrator.y[17]])      #body z-direction
            self.b2i = Rotation.from_matrix(b2imat)
            self.i2b = self.b2i.inv()
            self.time = integrator.t
            self.h=integrator.h_previous

            #Orientation - direction's of the body's coordinate system in the inertial frame
            xb = np.array([integrator.y[9],integrator.y[10],integrator.y[11]])   #body x-direction
            yb = np.array([integrator.y[12],integrator.y[13],integrator.y[14]])      #body y-direction
            zb = np.array([integrator.y[15],integrator.y[16],integrator.y[17]])

            new_row={"time":self.time,

                            "x_i":self.pos_i[0],
                            "y_i":self.pos_i[1],
                            "z_i":self.pos_i[2],

                            "w_bx":self.w_b[0],
                            "w_by":self.w_b[1],
                            "w_bz":self.w_b[2],

                            "vx_i":self.vel_i[0],
                            "vy_i":self.vel_i[1],
                            "vz_i":self.vel_i[2],

                            "xb":xb,
                            "yb":yb,
                            "zb":zb,
                            
                            "events":events}

            if verbose_log == True:
                launch_position = pos_i2l(self.pos_i,self.launch_site,self.time)
                launch_velocity = vel_i2l(self.vel_i,self.launch_site,self.time)
                w_b = self.w_b

                #Orientation
                x_b_i = self.b2i.apply([1,0,0])
                x_b_l = direction_i2l(x_b_i, self.launch_site, self.time)
                ypr = self.b2i.as_euler('zyx')

                #Aero forces aero_forces
                aero_forces, cop = self.aero_forces(self.b2i, self.pos_i, self.vel_i, self.time)
                aero_forces_l = direction_i2l(self.b2i.apply(aero_forces), self.launch_site, self.time)

                #Accelerations
                lin_acc, wdot_b = self.accelerations(self.pos_i, self.vel_i, self.w_b, self.b2i, self.time)

                #Centre of gravity
                cog = self.mass_model.cog(self.time)    
                verbose_info={"h":self.h,

                        "x_l":launch_position[0],
                        "y_l":launch_position[1],
                        "z_l":launch_position[2],
                        "vx_l":launch_velocity[0],
                        "vy_l":launch_velocity[1],
                        "vz_l":launch_velocity[2],

                        "yaw":ypr[0],
                        "pitch":ypr[1],
                        "roll":ypr[2],
                        "attitude_xi":x_b_i[0],
                        "attitude_yi":x_b_i[1],
                        "attitude_zi":x_b_i[2],
                        "attitude_xl":x_b_l[0],
                        "attitude_yl":x_b_l[1],
                        "attitude_zl":x_b_l[2],

                        "aero_xb":aero_forces[0],
                        "aero_yb":aero_forces[1],
                        "aero_zb":aero_forces[2],
                        "aero_xl":aero_forces_l[0],
                        "aero_yl":aero_forces_l[1],
                        "aero_zl":aero_forces_l[2],
                        "cop":cop,

                        "wdot_bx":wdot_b[0],
                        "wdot_by":wdot_b[1],
                        "wdot_bz":wdot_b[2],
                        
                        "cog": cog}
                new_row.update(verbose_info)

            record=record.append(new_row, ignore_index=True)
            if d==1000:
                print("t={:.2f} s alt={:.2f} km (h={} s)".format(self.time, self.altitude(self.pos_i)/1000, integrator.h_abs))
                d=0
            c+=1
            d+=1
        return record

    def check_phase(self):
        events=[]
        if self.on_rail==True:
            #Remember that the launch site's origin is at altitude=0, so the rocket's initial position is [0, 0, self.launch_site.alt] in launch site coordinates
            flight_distance = np.linalg.norm(pos_i2l(self.pos_i, self.launch_site, self.time) - np.array([0, 0, self.launch_site.alt]))  

            if flight_distance>=self.launch_site.rail_length:            

                print("Cleared rail at t={:.2f} s with alt={:.2f} m and TtW={:.2f}".format(self.time,
                self.altitude(self.pos_i),
                np.linalg.norm(self.accelerations(self.pos_i, self.vel_i, self.w_b, self.b2i, self.time)[0])/9.81)
                )
                self.on_rail=False
                events.append("Cleared rail")
        return events


#pos_l2i and pos_i2l HAVE BEEN CHANGED BUT HAS NOT BEEN TESTED        
def pos_l2i(pos_l, launch_site,time):
    """Converts position in launch frame to position in inertial frame
    Args:
        position (Numpy Array): Position in launch frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in inertial frame
    """
    #Converting spherical coordinates to Cartesian:
    #https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).

    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]
    
    #print("r_launch_site - r_earth = {}".format(np.linalg.norm(pos_launch_site_i) - r_earth))
    #print("pos_l = {}".format(pos_l))
    #print("direction_l2i(pos_l) = {}".format(direction_l2i(pos_l, launch_site, time)))

    pos_rocket_i = pos_launch_site_i + direction_l2i(pos_l, launch_site, time)

    return pos_rocket_i

def pos_i2l(position,launch_site,time):
    """Converts position in inertial frame to position in launch frame
    Args:
        position (Numpy Array): Position in inertial frame
        launch_site (LaunchSite): The relivant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in launch frame
    """
    #Converting spherical coordinates to Cartesian:
    #https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).

    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    pos_rocket_i = position
    pos_rocket_l =  direction_i2l(pos_rocket_i - pos_launch_site_i, launch_site, time)

    return pos_rocket_l

def vel_i2l(vel_i, launch_site, time):
    """Converts inertial velocity to velocity in launch frame
    Args:
        velocity (np.array): [x,y,z] Velocity in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in launch frame
    """    
    
    w_earth = np.array([0, 0, ang_vel_earth])
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                    r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                    r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    # v = w x r for a rigid body, where v, w and r are vectors
    launch_site_velocity_i = np.cross(w_earth, pos_launch_site_i)

    return direction_i2l(vel_i - launch_site_velocity_i, launch_site, time) 

def vel_l2i(vel_l, launch_site, time):
    """Converts launch frame velocity to velocity in inertial frame
    Args:
        velocity (Numpy array): [x,y,z] velocity in launch frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in inertial frame
    """    
    w_earth = np.array([0, 0, ang_vel_earth])
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    # v = w x r for a rigid body, where v, w and r are vectors
    launch_site_velocity_i = np.cross(w_earth, pos_launch_site_i)

    return direction_l2i(vel_l, launch_site, time) + launch_site_velocity_i


def direction_i2l(vector, launch_site, time):
    """Converts inertial direction vector to a direction vector in launch frame
    Args:
        vector (np.array): [x,y,z] vector in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Vector in launch frame
    """   

    #I have no idea why we need the negatives in front of the yaw and pitch - but it seems to give the right answers
    return Rotation.from_euler('zy', [-launch_site.longi - (180/np.pi)*ang_vel_earth*time, -90 + launch_site.lat], degrees=True).apply(vector)

def direction_l2i(vector, launch_site, time):
    """Converts launch direction vector to a direction vector in inertial frame
    Args:
        vector (np.array): [x,y,z] vector in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Vector in launch frame
    """    
    #I have no idea why we need the negatives in front of the yaw and pitch
    return Rotation.from_euler('zy', [-launch_site.longi - (180/np.pi)*ang_vel_earth*time, -90 + launch_site.lat], degrees=True).inv().apply(vector)


def rot_matrix(ypr, inverse=False):
    """Generates a rotation matrix between frames which are rotated by yaw, pitch and roll specified by orientation
    Left hand matrix multiply this by the relivant vector (i.e. np.matmul(rotation_matrix(....),vec))

    Args:
        orientation (np.array): The rotation of the frames relative to eachother as yaw, pitch and roll (about z, about y, about x)
        inverse (bool, optional): If the inverse is required (i.e. when transforming from the frame which is rotated by orientation). Defaults to False.

    Returns:
        [type]: [description]
    """    

    rot = Rotation.from_euler('zyx', ypr)

    if inverse==True:
        rot = rot.inv()

    return rot.as_matrix()
