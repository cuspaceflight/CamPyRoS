"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
COORDINATE SYSTEM NOMENCLATURE
x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
x_i,y_i,z_i = Inertial coordinate system (does not rotate, origin at centre of the Earth)
x_l, y_l, z_l = Launch site coordinate system (origin on launch site, rotates with the Earth)

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

import csv
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation
import pandas as pd


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

r_earth = 6378137 #(earth's semimarjor axis in meters)
#e_earth = 0.081819191 #earth ecentricity
e_earth = 0 #for simplicity of other calculations for now - if changed need to update the launchsite orientation and velocity transforms
ang_vel_earth=7.292115090e-5 #rads / s

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
        
        #Initialise the rocket's orientation - store it in a scipy.spatial.transform.Rotation object 
        #Remember that the body's x-direction will be in the launch frame's z-direction
        x = direction_l2i([0,0,1], self.launch_site, self.time)     #x = where the rocket's 'x' points in the inertial frame
        y = direction_l2i([0,1,0], self.launch_site, self.time)     #y for the body is aligned with y for the launch site
        z = direction_l2i([1,0,0], self.launch_site, self.time)     #z = where the rocket's 'z' points in the inertial frame

        mat_b2i = np.zeros([3,3])
        mat_b2i[:,0] = x
        mat_b2i[:,1] = y
        mat_b2i[:,2] = z
        self.b2i = Rotation.from_matrix(mat_b2i)                            

        #Get the additional bit due to the angling of the rail
        rail_rotation = Rotation.from_euler('zy', [self.launch_site.rail_yaw, self.launch_site.rail_pitch], degrees=True)
        self.b2i = rail_rotation*self.b2i       #Body-to-Inertial Rotation - you can apply it to a vector with self.b2i.apply(vector)
        self.i2b = self.b2i.inv()               #Inertial-to-Body Rotation

        #Initialise angular positions and angular velocities
        self.pos_i = pos_l2i(np.array([0,0,0]),launch_site,0)                        #Position in inertial coordinates
        self.vel_i = vel_l2i([0,0,0],launch_site.lat, launch_site.longi,0,0)         #Velocity in intertial coordinates

        self.w_b = ([0,0,0])                                                         #Angular velocity in body coordinates
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
        wind_inertial = vel_l2i(self.launch_site.wind, np.angle(1j*pos_i[2] + np.sqrt(pos_i[0]**2+pos_i[1]**2)) *180/np.pi, np.angle(1j* round(pos_i[0],10) + round(pos_i[1],10))*180/np.pi, time, alt)
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
        
        #Calculate moments in body coordinates using moment = r x F    
        aero_moment_b = np.cross(r_cop_cog_b, aero_force_b)
        thrust_moment_b = np.cross(r_engine_cog_b, thrust_b)
        
        #Convert forces to inertial coordinates
        thrust = b2i.apply(thrust_b)
        aero_force = b2i.apply(aero_force_b)
        
        #Get total force and moment
        F = thrust + aero_force + self.gravity(time, pos_i)
        Q_b = aero_moment_b + thrust_moment_b   
        
        #Calculate angular velocities using Euler's equations - IIA Engineering, Module 3C5, Rigid body dynamics handout (page 18)
        i_b = np.array([self.mass_model.ixx(time),
                        self.mass_model.iyy(time),
                        self.mass_model.izz(time)])     #Moments of inertia [ixx, iyy, izz]

        wdot_b = np.array([(Q_b[0] + (i_b[1] - i_b[2])*w_b[1]*w_b[2]) / i_b[0]
                            ,(Q_b[1] + (i_b[2] - i_b[0])*w_b[2]*w_b[0]) / i_b[1]
                            ,(Q_b[2] + (i_b[0] - i_b[1])*w_b[0]*w_b[1]) / i_b[2]])
    
        if self.on_rail==True:
            F=np.array([F[0],0,0])
            wdot_b=np.array([wdot_b[0],0,0])
        
        #F = ma in inertial coordinates
        lin_acc = F/self.mass_model.mass(time)

        if abs(wdot_b[0]) > 3:
            wdot_b[0] = 3*np.sign(wdot_b[0])
            
        if abs(wdot_b[1]) > 3:
            wdot_b[1] = 3*np.sign(wdot_b[1])
            
        if abs(wdot_b[2]) > 3:
            wdot_b[2] = 3*np.sign(wdot_b[2])
        
        return np.stack([lin_acc, wdot_b])

    def fdot(self, f, time):
        '''
        f contains everything needed to full define the rocket
        'fdot' here is the same as 'ydot' in the 2P1 (2nd Year) Engineering Lagrangian dynamics notes RK4 section

        f = [pos_i, vel_i, w_b, xb, yb, zb]
        fdot = [vel_i, acc_i, w_bdot, xbdot, ybdot, zbdot]

        This returns fdot
        '''

        pos_i = f[0]
        vel_i = f[1]
        w_b = f[2]
        xb = f[3]
        yb = f[4]
        zb = f[5]
        
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

        return np.array([vel_i, acc_i, w_bdot, xbdot, ybdot, zbdot])

    def step_RK4(self):
        '''Runga Kutta 4th order method for integration'''
        
        #f = [pos_i, vel_i, w_b, xb, yb, zb]                               <--- Everything needed to fully define the state of the rocket
        #fdot = [vel_i, acc_i, w_bdot, xbdot, ybdot, zbdot]

        #xb = body's x-direction in inertial, yb = body's y-direction in inertial, etc...
        xb = self.b2i.as_matrix()[:,0]
        yb = self.b2i.as_matrix()[:,1]
        zb = self.b2i.as_matrix()[:,2]

        fn = [self.pos_i, self.vel_i, self.w_b, xb, yb, zb]

        dt = self.h
        t = self.time
        
        k1 = self.fdot(fn, t)
        k2 = self.fdot(fn + k1*dt/2, t + dt/2)
        k3 = self.fdot(fn + k2*dt/2, t + dt/2)
        k4 = self.fdot(fn + k3*dt, t + dt)

        fnplusone = fn + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*dt   # + O(dt^5)
        
        #Update rocket's status
        self.pos_i = fnplusone[0]
        self.vel_i = fnplusone[1]
        self.w_b = fnplusone[2]
        b2imat = np.zeros([3,3])
        b2imat[:,0] = fnplusone[3]      #body x-direction
        b2imat[:,1] = fnplusone[4]      #body y-direction
        b2imat[:,2] = fnplusone[5]      #body z-direction
        self.b2i = Rotation.from_matrix(b2imat)
        self.i2b = self.b2i.inv()

        #Step forwards time
        self.time = self.time + dt

    def step_euler(self):
        """Semi-implicit Euler method for integration"""

        pos_i = self.pos_i
        vel_i = self.vel_i
        w_b = self.w_b

        b2i = self.b2i
        i2b = self.i2b

        time = self.time
        
        #Integrate translational position and velocity
        self.vel_i = vel_i + self.accelerations(pos_i, vel_i, w_b, b2i, time)[0] * self.h
        self.pos_i = pos_i + self.vel_i * self.h
        
        #Integrate angular acceleration
        wdot_b = self.accelerations(pos_i, vel_i, w_b, b2i, time)[1]
        self.w_b = w_b + wdot_b * self.h

        #Integrate angular velocity to get orientation
        b2imatdot = self.w_b_to_b2imatdot(w_b, b2i)
        self.b2i = Rotation.from_matrix(b2i.as_matrix() + b2imatdot*self.h)           
        self.i2b = self.b2i.inv()

        #Step time
        self.time+=self.h
        
        #WARNING: I suspect that errors can accumulate if the columns of b2i.as_matrix() deviate from being unit vectors
        #Uncommenting the lines below will print out the norm of each column of b2i.as_matrix()
        #From previous experience, they seem to be stable at around 1.
        #print("[{}, {}, {}]".format(np.linalg.norm(b2i.as_matrix()[:,0]), np.linalg.norm(b2i.as_matrix()[:,1]), np.linalg.norm(b2i.as_matrix()[:,2])))

    def check_phase(self):
        if self.on_rail==True:
            flight_distance = np.linalg.norm(pos_i2l(self.pos_i,self.launch_site,self.time))
            if flight_distance>=self.launch_site.rail_length:
                print("Cleared rail at t={:.2f} s with alt={:.2f} m and TtW={:.2f}".format(self.time,
                self.altitude(self.pos_i),
                np.linalg.norm(self.accelerations(self.pos_i, self.vel_i, self.w_b, self.b2i, self.time)[0])/9.81)
                )
                self.on_rail=False

#pos_l2i and pos_i2l HAVE BEEN CHANGED BUT HAS NOT BEEN TESTED        
def pos_l2i(position,launch_site,time):
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

    pos_rocket_l = position
    pos_rocket_i = pos_launch_site_i + direction_l2i(pos_rocket_l, launch_site, time)

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

#Where is this used?? ("inertial_to_inertial_long_lat")
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

def vel_i2l(velocity, launch_site, time):
    """Converts inertial velocity to velocity in launch frame
    Args:
        velocity (np.array): [x,y,z] Velocity in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in launch frame
    """    
    launch_site_velocity = np.array([0,ang_vel_earth*(r_earth+launch_site.alt)*np.cos(launch_site.lat*np.pi/180),0])
    inertial_rot_launch = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,launch_site.lat*np.pi/180+np.pi/2,0],True),velocity)
    return inertial_rot_launch-launch_site_velocity

def vel_l2i(velocity, launch_site_lat, launch_site_longi, time, alt):
    """Converts launch frame velocity to velocity in inertial frame
    Args:
        velocity (Numpy array): [x,y,z] velocity in launch frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Velocity in inertial frame
    """    
    launch_site_velocity = np.array([0,ang_vel_earth*(r_earth+alt)*np.cos(launch_site_lat*np.pi/180),0])
    launch_rot_inertial = np.matmul(rot_matrix([time*ang_vel_earth+launch_site_longi*np.pi/180,0,0]),velocity)

    return launch_rot_inertial+launch_site_velocity

def direction_i2l(vector, launch_site, time):
    """Converts inertial direction vector to a direction vector in launch frame
    Args:
        vector (np.array): [x,y,z] vector in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Vector in launch frame
    """    
    inertial_rot_launch = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,launch_site.lat*np.pi/180+np.pi/2,0],True),vector)
    return inertial_rot_launch

def direction_l2i(vector, launch_site, time):
    """Converts launch direction vector to a direction vector in inertial frame
    Args:
        vector (np.array): [x,y,z] vector in inertial frame
        launch_site (LaunchSite): The relevant launch site
        time (float): Elapsed time from ignition
    Returns:
        Numpy array: Vector in launch frame
    """    
    inertial_rot_launch = np.matmul(rot_matrix([time*ang_vel_earth+launch_site.longi*np.pi/180,launch_site.lat*np.pi/180+np.pi/2,0],False),vector)
    return inertial_rot_launch

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

def run_simulation_RK4(rocket, max_time=200):
    c=0
    d=0
    """
    Runs the simulation and outputs everything that is needed to fully define the rocket at each time step.
    Everything else can be calculated from this data.

    Args:
        rocket (Rocket): The rocket to be simulated

    Returns:
        Pandas Dataframe: Record of position, velocity and mass at time t 
    """  
    print("Running simulation")
    record=pd.DataFrame({})
    while (rocket.altitude(rocket.pos_i)>=0 and rocket.time<max_time):
        rocket.check_phase()
        rocket.step_RK4()

        #Orientation - direction's of the body's coordinate system in the inertial frame
        b2imatrix = rocket.b2i.as_matrix()
        xb = b2imatrix[:,0]
        yb = b2imatrix[:,1]
        zb = b2imatrix[:,2]

        new_row={"time":rocket.time,

                        "x_i":rocket.pos_i[0],
                        "y_i":rocket.pos_i[1],
                        "z_i":rocket.pos_i[2],

                        "w_bx":rocket.w_b[0],
                        "w_by":rocket.w_b[1],
                        "w_bz":rocket.w_b[2],

                        "vx_i":rocket.vel_i[0],
                        "vy_i":rocket.vel_i[1],
                        "vz_i":rocket.vel_i[2],

                        "xb":xb,
                        "yb":yb,
                        "zb":zb}
        record=record.append(new_row, ignore_index=True)
        if d==1000:
            print("t={:.2f} s alt={:.2f} km (h={} s)".format(rocket.time, rocket.altitude(rocket.pos_i)/1000, rocket.h))
            d=0

        c+=1
        d+=1
    return record

def run_simulation_RK4_debug(rocket, max_time=200):
    c=0
    d=0
    """Runs the simulation but produces lots of extra data in-situ that would be useful for debugging

    Args:
        rocket (Rocket): The rocket to be simulated

    Returns:
        Pandas Dataframe: Record of position, velocity and mass at time t 
    """  
    print("Running simulation")
    record=pd.DataFrame({})
    while (rocket.altitude(rocket.pos_i)>=0 and rocket.time<max_time):
        rocket.check_phase()
        rocket.step_RK4()

        #Position and velocity
        launch_position = pos_i2l(rocket.pos_i,rocket.launch_site,rocket.time)
        launch_velocity = vel_i2l(rocket.vel_i,rocket.launch_site,rocket.time)
        w_b = rocket.w_b

        #Orientation
        x_b_i = rocket.b2i.apply([1,0,0])
        x_b_l = direction_i2l(x_b_i, rocket.launch_site, rocket.time)
        ypr = rocket.b2i.as_euler('zyx')

        #Aero forces aero_forces
        aero_forces, cop = rocket.aero_forces(rocket.b2i, rocket.pos_i, rocket.vel_i, rocket.time)
        aero_forces_l = direction_i2l(rocket.b2i.apply(aero_forces), rocket.launch_site, rocket.time)

        #Accelerations
        lin_acc, wdot_b = rocket.accelerations(rocket.pos_i, rocket.vel_i, rocket.w_b, rocket.b2i, rocket.time)

        #Centre of gravity
        cog = rocket.mass_model.cog(rocket.time)

        burnout_time = rocket.motor.motor_time_data[-1]
        new_row={"time":rocket.time,
                        "burnout_time":burnout_time,
                        "h":rocket.h,

                        "x_i":rocket.pos_i[0],
                        "y_i":rocket.pos_i[1],
                        "z_i":rocket.pos_i[2],
                        "x_l":launch_position[0],
                        "y_l":launch_position[1],
                        "z_l":launch_position[2],
                        "w_bx":w_b[0],
                        "w_by":w_b[1],
                        "w_bz":w_b[2],
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
        record=record.append(new_row, ignore_index=True)
        if d==1000:
            print("t={:.2f} s alt={:.2f} km (h={} s)".format(rocket.time, rocket.altitude(rocket.pos_i)/1000, rocket.h))
            d=0
        #print("alt={:.0f} time={:.1f}".format(rocket.altitude(rocket.pos), rocket.time))

        c+=1
        d+=1
    return record

def run_simulation_euler(rocket, max_time=200):
    c=0
    d=0
    """Runs the simulaiton to completeion outputting dictionary of the position, velocity and mass of the rocket

    Args:
        rocket (Rocket): The rocket to be simulated

    Returns:
        Pandas Dataframe: Record of position, velocity and mass at time t 
    """  
    print("Running simulation")
    record=pd.DataFrame({"time":[],"x":[],"y":[],"z":[],"v_x":[],"v_y":[],"v_z":[]}) #time:[position,velocity,mass]
    while (rocket.altitude(rocket.pos_i)>=0 and rocket.time<max_time):
        rocket.check_phase()
        rocket.step_euler()

        #Position and velocity
        launch_position = pos_i2l(rocket.pos_i,rocket.launch_site,rocket.time)
        launch_velocity = vel_i2l(rocket.vel_i,rocket.launch_site,rocket.time)
        w_b = rocket.w_b

        #Orientation
        x_b_i = rocket.b2i.apply([1,0,0])
        x_b_l = direction_i2l(x_b_i, rocket.launch_site, rocket.time)
        ypr = rocket.b2i.as_euler('zyx')

        #Aero forces aero_forces
        aero_forces, cop = rocket.aero_forces(rocket.b2i, rocket.pos_i, rocket.vel_i, rocket.time)
        aero_forces_l = direction_i2l(rocket.b2i.apply(aero_forces), rocket.launch_site, rocket.time)

        #Accelerations
        lin_acc, wdot_b = rocket.accelerations(rocket.pos_i, rocket.vel_i, rocket.w_b, rocket.b2i, rocket.time)

        #Centre of gravity
        cog = rocket.mass_model.cog(rocket.time)

        burnout_time = rocket.motor.motor_time_data[-1]
        new_row={"time":rocket.time,
                        "burnout_time":burnout_time,
                        "h":rocket.h,

                        "x_i":rocket.pos_i[0],
                        "y_i":rocket.pos_i[1],
                        "z_i":rocket.pos_i[2],
                        "x_l":launch_position[0],
                        "y_l":launch_position[1],
                        "z_l":launch_position[2],
                        "w_bx":w_b[0],
                        "w_by":w_b[1],
                        "w_bz":w_b[2],
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
        record=record.append(new_row, ignore_index=True)
        if d==1000:
            print("t={:.2f} s alt={:.2f} km (h={} s)".format(rocket.time, rocket.altitude(rocket.pos_i)/1000, rocket.h))
            d=0
        #print("alt={:.0f} time={:.1f}".format(rocket.altitude(rocket.pos), rocket.time))

        c+=1
        d+=1
    return record







#Legacy functions
'''
def quatmul(q1, q2):
    #eturns q1 x q2, as a 1x4 Numpy array
    #HAS BEEN TESTED - WORKS CORRECTLY

    q1 = np.array(q1)
    q2 = np.array(q2)

    if q1.shape[0] != 4 or q1.shape[0] != 4:
        raise ValueError("Invalid shape for quatmul, must be [qs, qx, qy, qz].")

    else:
        s = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
        v = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
        return np.array([s, v[0], v[1], v[2]])

def quatnorm(q):
    #Returns the norm of the quaternion
    if q.shape[0] != 4:
        raise ValueError("Invalid shape for quatnorm, must be [qs, qx, qy, qz].")
    else:
        return (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5

def quatinv(q):
    #Returns inverse of q as a 1x4 Numpy array
    if q.shape[0] != 4:
        raise ValueError("Invalid shape for quatinv, must be [qs, qx, qy, qz].")

    elif quatnorm(q) == 0:
        raise ValueError("Cannot inverse a quaternion with zero norm")

    else:
        q_star = np.array([q[0], -q[1], -q[2], -q[3]])
        return q_star/quatnorm(q_star)


#Legacy functions from inside the Rocket object

def w_b_to_quat_i2bdot(self, w_b, i2b):
    
    #Legacy function. Converts angular velocity in the body frame to the rate of change of i2b
    #Returns i2bdot [qsdot, qxdot, qydot, qzdot]
    
    #https://math.stackexchange.com/questions/773902/integrating-body-angular-velocity
    #print("Warning: w_b_to_quat_i2bdot() has not been tested properly, and is likely to give inaccurate/wrong answers")
    return 0.5 * quatmul(i2b.as_quat(), np.array([0, w_b[0], w_b[1], w_b[2]]))

def w_b_to_yprdot(self, w_b, ypr):
    
    #Legacy function. Converts angular velocity in the body frame to yaw-pitch-roll rates
    #Returns yprdot = [yawdot, pitchdot, rolldot]
    
    phi = ypr[0]
    theta = ypr[1]

    B = [[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]]

    print("Warning: w_b_to_yprdot() has not been tested properly, and is likely to give inaccurate/wrong answers")
    return np.matmul(B, w_b)    

    def w_b_to_b2imatdot(self, w_b, b2i):
        
        #Returns the rate of change of the b2i matrix with time
        #You could then integrate it by doing, e.g.

        #b2i.as_matrix() = b2i.as_matrix() + matdot*dt
        
        x = self.b2i.as_matrix()[:,0]
        y = self.b2i.as_matrix()[:,1]
        z = self.b2i.as_matrix()[:,2]

        w_i = self.b2i.apply(w_b)

        xdot = np.cross(w_i, x)
        ydot = np.cross(w_i, y)
        zdot = np.cross(w_i, z)

        matdot = np.zeros([3,3])
        matdot[:,0] = xdot
        matdot[:,1] = ydot
        matdot[:,2] = zdot

        return matdot


'''

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

