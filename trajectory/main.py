"""6DOF Trajectory Simulator
Contains the classes and functions for the core trajectory simulation
Example
-------
A small, single stage rocket can be found in examples, to run
        $ python example/example.py
Notes
-----
    SI units unless stated otherwise
    Coordinate systems:
    x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
    x_i,y_i,z_i = Inertial coordinate system (does not rotate, origin at centre of the Earth)
    x_l, y_l, z_l = Launch site coordinate system (origin has the launch site's longitude and latitude, but is at altitude = 0). Rotates with the Earth.
    Directions are defined below.
    - Body:
        y points east and z north at take off (before rail alignment is accounted for) x up.
        x is along the "long" axis of the rocket.
    - Launch site:
        z points perpendicular to the earth, y in the east direction and x tangential to the earth pointing south
        
    - Inertial:
        Origin at centre of the Earth
        z points to north from centre of earth, x aligned with launchsite at start and y orthogonal
Attributes
----------
r_earth : float
    Radius of Earth/m
e_earth : float
    Eccentricity of the Earth, currently set to zero to simplify calculations (i.e. spherical Earth model is being used)
ang_vel_earth : float
    The angular velocity of the Earth
"""

import csv, warnings, os, sys, json, iris, requests, metpy.calc, os.path
from metpy.units import units
import numpy as np
import pandas as pd

import scipy.interpolate
from scipy.spatial.transform import Rotation
import scipy.integrate as integrate
from datetime import date

from .constants import r_earth, ang_vel_earth, f
from .transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, i2lla
from ambiance import Atmosphere

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '\n%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line

class Wind:
    #Data will be strored in data_loc in the format lat_long_date_run_period.grb2 where lat and long are the bottom left values
    #Run has to be 00, 06, 12 or 18

    def __init__(self,initial_long,initial_lat,variable=True,default=np.array([0,0,0]),data_loc="data/wind/gfs",run_date=date.today().strftime("%Y%m%d"),forcast_time="00",forcast_plus_time="000"):
        warnings.warn("Wind data becomes problamatic at the poles and when integer long and lat are given")
        self.centre_lat=initial_lat
        self.centre_long=initial_long
        self.data_loc=data_loc#must be in last week for now
        self.variable = variable
        self.default=default

        if variable == True:
            if forcast_time not in ["00","06","12","18"]:
                warning.warn("The forcast run selected is not valid, must be '00', '06', '12' or '18'. This will be fatal on file load")
            valid_times=["00%s"%n for n in range(0,10)]+["0%s"%n for n in range(10,100)]+["%s"%n for n in range(100,385)]
            if forcast_plus_time not in valid_times:
                warning.warn("The forcast time selected is not valid, must be three digit string time between 000 and 384. Thi siwll be fatal on file load")
            self.date=run_date
            self.forcast_time=forcast_time
            self.run_time=forcast_plus_time
            self.df,self.lats,self.longs=self.load_data(self.centre_lat,self.centre_long)

    def load_data(self,lat,longi):
        lat_top=round(lat/.25)*.25
        if lat_top>lat:
            lat_bottom=lat_top-.25
        else:
            lat_bottom=lat_top+.25
            lat_top,lat_bottom=lat_bottom,lat_top
            
        long_left=round(longi/.25)*.25
        if long_left<longi:
            long_right=long_left+.25
        else:
            long_right=long_left-.25
            long_left,long_right=long_right,long_left

        if not os.path.isfile("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time)):
            #This does download 3 rows that aren't needed but I can't work out how to yeet them
            print("Downloading files")
            url="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t{run}z.pgrb2.0p25.f{hour}&lev_0.4_mb=on&lev_1000_mb=on&lev_100_mb=on&lev_10_mb=on&lev_150_mb=on&lev_15_mb=on&lev_180-0_mb_above_ground=on&lev_1_mb=on&lev_200_mb=on&lev_20_mb=on&lev_250_mb=on&lev_255-0_mb_above_ground=on&lev_2_mb=on&lev_300_mb=on&lev_30-0_mb_above_ground=on&lev_30_mb=on&lev_350_mb=on&lev_3_mb=on&lev_400_mb=on&lev_40_mb=on&lev_450_mb=on&lev_500_mb=on&lev_50_mb=on&lev_550_mb=on&lev_5_mb=on&lev_600_mb=on&lev_650_mb=on&lev_700_mb=on&lev_70_mb=on&lev_750_mb=on&lev_7_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&var_HGT=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon={leftlon}&rightlon={rightlon}&toplat={toplat}&bottomlat={bottomlat}&dir=%2Fgfs.{date}%2F{run}".format(leftlon=long_left,rightlon=long_right,toplat=lat_top,bottomlat=lat_bottom,date=self.date,run=self.forcast_time,hour=self.run_time)
            
            r = requests.get(url, stream=True)

            with open("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time),'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: 
                        f.write(chunk)

        if os.path.getsize("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time))<1000:
            raise RuntimeError("The weather data you requested was not found, this is usually because it was for an invalid date/time")

        print("Loading wind data for N%s E%s for %sh after forcast run at %s on %s"%(lat,longi,self.run_time,self.forcast_time,self.date))
        data=iris.load("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time))
        lats=list(data[0].coord("latitude").points)
        longs=list(data[0].coord("longitude").points)
        df=pd.DataFrame(columns=["lat","long","alt","w_x","w_y"])
        for long in longs:
            for lat in lats:
                press=data[1].extract(iris.Constraint(latitude=lat,longitude=long)).coord("pressure").points
                for pres in press:
                    w_x=data[1].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data
                    w_y=data[3].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data
                    alt=10*metpy.calc.geopotential_to_height(data[0].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data*units.m**2/units.s**2).magnitude
                    row={"lat":lat,"long":long,"alt":alt,"w_x":w_x,"w_y":w_y}
                    df=df.append(row,ignore_index=True)

        return df,lats,longs

    def get_wind(self,lat,long,alt):
        if alt<-1000:
            #This stops the weird yeeting through the earth in the last step
            lat,long,alt=self.centre_lat,self.centre_long,0
        elif alt>80000:
            lat,long,alt=self.centre_lat,self.centre_long,max(self.df["alt"])
        if self.variable == True:
            lat_query=[None,None]
            lat_query[0]=round(lat/.25)*.25
            if lat_query[0]>lat:
                lat_query[1]=lat_query[0]-.25
            else:
                lat_query[1]=lat_query[0]+.25
                lat_query[0],lat_query[1]=lat_query[1],lat_query[0]
            long_query=[None,None]
            long_query[0]=round(long/.25)*.25
            if long_query[0]<long:
                long_query[1]=long_query[0]+.25
            else:
                long_query[1]=long_query[0]-.25
                long_query[0],long_query[1]=long_query[1],long_query[0]

            available=False
            while available==False:
                #I don't think this is the most efficient but *should* work
                if not (min(self.lats)<lat<max(self.lats) and min(self.longs)<long<max(self.longs)).all():
                    new_df,lats,longs=self.load_data(lat,long)
                    self.df=self.df.append(new_df)
                    self.lats+=lats
                    self.longs+=longs
                else:
                    available=True

            points = [[None,None],[None,None]]
            for n in [0,1]:
                points[n][0]=self.df.query('lat == %s'%lat_query[n])
                points[n][1]=self.df.query('lat == %s'%lat_query[n])
                for m in [0,1]:
                    points[n][m]=points[n][m].query('long == %s'%long_query[m])
                    
            meaning = [[None,None],[None,None]]
            for n in [0,1]:
                meaning[n][0]="lat_%s"%n
                meaning[n][1]="lat_%s"%n
                for m in [0,1]:
                    meaning[n][m]+=", long_%s"%m
                    
                    
            wind_x = [[None,None],[None,None]]
            wind_y = [[None,None],[None,None]]
            for n in [0,1]:
                for m in [0,1]:
                    if alt > min(points[n][m]["alt"]) and alt < max(points[n][m]["alt"]):
                        wind_x[n][m]=scipy.interpolate.interp1d(points[n][m]["alt"],points[n][m]["w_x"])(alt)
                        wind_y[n][m]=scipy.interpolate.interp1d(points[n][m]["alt"],points[n][m]["w_y"])(alt)
                    else:
                        wind=points[n][m].iloc[(points[n][m]["alt"]-alt).abs().argsort()[:1]]
                        wind_x[n][m],wind_y[n][m]=wind["w_x"].astype(float).to_numpy()[0],wind["w_y"].astype(float).to_numpy()[0]
                    
            wind_lats_x = [None,None]
            wind_lats_y = [None,None]
            for n in [0,1]:
                wind_lats_x[n]=scipy.interpolate.interp1d(long_query,wind_x[n])(long)
                wind_lats_y[n]=scipy.interpolate.interp1d(long_query,wind_y[n])(long)
            
            wind_x=scipy.interpolate.interp1d(lat_query,wind_lats_x)(lat)
            wind_y=scipy.interpolate.interp1d(lat_query,wind_lats_y)(lat)

            #Until this moment I thought long was a key word since VSCode always hilights it, I realise its not now and apologise for using longi instead when it was unneccasary
            #I think x is east and y is north, not confirmed, should be checked
            return np.array([-wind_y,wind_x,0])
        else:
            return self.default
class Parachute:
    """Object holding the parachute information
    Note
    ----
    The parachute model does not currently simulate the true orientation of the rocket instead it orientates
    it such that it faces back first into the wind (as intuition would suggest).
    This is due to problems trying to impliment the chute exerting torque on the body, possibly because it has to flip 
    the rocket over at apogee
    Parameters
    ----------
    main_s : float
        Area of main chute/m^2
    main_c_d : float
        Coefficient of drag for main chute/
    drogue_s : float
        Area of main chute/m^2
    drogue_c_d : float
        Coefficient of drag for main chute/
    main_alt : float
        Altitude at which main deploys
    attatch_distance : float
        Distance from the nose of the rocket that the parachute is attatched /m
    Attributes
    ----------
    main_s : float
        Area of main chute/m^2
    main_c_d : float
        Coefficient of drag for main chute/
    drogue_s : float
        Area of main chute/m^2
    drogue_c_d : float
        Coefficient of drag for main chute/
    main_alt : float
        Altitude at which main deploys
    attatch_distance : float
        Distance from the nose of the rocket that the parachute is attatched /m
    """
    def __init__(self,main_s,main_c_d,drogue_s,drogue_c_d,main_alt,attatch_distance):
        self.main_s=main_s
        self.main_c_d=main_c_d
        self.drogue_s=drogue_s
        self.drogue_c_d=drogue_c_d

        self.main_alt = main_alt
        self.attatch_distance=attatch_distance

    def get(self,alt):
        """Returns the current parachute area and drag coefficient (checks if main is deployed)
        Parameters
        ----------
        alt : float
            Rocket altitude /m
        Returns
        -------
        float
            Drag coefficient /
            Area /m^2
        """ 
        if alt<self.main_alt:
            c_d=self.main_c_d
            s=self.main_s
        else:
            c_d=self.drogue_c_d
            s=self.drogue_s
        return c_d,s

class Motor:
    """Object holding the performance data for the engine
    Parameters
    ----------
    motor_time_data : list
        Time since ignition (with times corresponding to the other input lists) /s
    prop_mass_data : list
        Propellant mass /kg
    cham_pres_data : list
        Chamber Pressure /Pa
    throat_data : list
        Throat diameter /m
    gamma_data : list
        Nozzle inlet gamma (ratio of specific heats)
    nozzle_efficiency_data : list
        Nozzle efficiency
    exit_pres_data : list 
        Exit pressure /Pa
    area_ratio_data : list 
        Area ratio
    Attributes
    ----------
    motor_time_data : list
        Time since ignition (with times corresponding to the other input lists) /s
    prop_mass_data : list
        Propellant mass /kg
    cham_pres_data : list
        Chamber Pressure /Pa
    throat_data : list
        Throat diameter /m
    gamma_data : list
        Nozzle inlet gamma (ratio of specific heats)
    nozzle_efficiency_data : list
        Nozzle efficiency
    exit_pres_data : list 
        Exit pressure /Pa
    area_ratio_data : list 
        Area ratio
    """   
      
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

class LaunchSite:
    """Object holding the launch site information
    Parameters
    ----------
    rail_length : float
        Length of launch rail /m
    rail_yaw : float
        Angle of rotation (using a right hand rule) about the launch site z-axis /degrees. Examples: rail_yaw = 0 points South, rail_yaw = 90 points East.
    rail_pitch : float
        Angle between the rail and the launch site z-axis (i.e. angle to the vertical) /degrees. Example: rail_pitch = 0 points up.
    alt : float
        Altitude /m
    longi : float
        Londditude /degrees
    lat : float
        Latitude /degrees
    wind : list, optional
        Wind vector at launch site. Defaults to [0,0,0]. Will increase completness/complexity at some point to include at least altitude variation.
    Attributes
    ----------
    rail_length : float
        Length of launch rail /m
    rail_yaw : float
        Angle of rotation about the z axis (north pointing) /rad
    rail_pitch : float
        Angle of rotation about "East" pointing y axis - in order to simplify calculations below this needs to be measured in the yaw then pitch order rad
    alt : float
        Altitude /m
    longi : float
        Londditude /degrees
    lat : float
        Latitude /degrees
    wind : list, optional
        Wind vector at launch site. Defaults to [0,0,0]. Will increase completness/complexity at some point to include at least altitude variation.
    """
    def __init__(self, rail_length, rail_yaw, rail_pitch, alt, longi, lat, variable_wind=True,default_wind=np.array([0,0,0]),wind_data_loc="data/wind/gfs",run_date=date.today().strftime("%Y%m%d"),forcast_time="00",forcast_plus_time="000"):
        self.rail_length = rail_length
        self.rail_yaw = rail_yaw
        self.rail_pitch = rail_pitch
        self.alt = alt
        self.longi = longi
        self.lat = lat
        self.wind = Wind(longi,lat,variable=variable_wind,default=default_wind,data_loc=wind_data_loc,run_date=run_date,forcast_time=forcast_time,forcast_plus_time=forcast_plus_time)
 
class RasAeroData: 
    """Object holding aerodynamic data from a RasAero II 'Aero Plots' export file
    Note
    ----
    Relies on an axially symetric body
    Parameters
    ----------
    file_location_string : string
        Location of RASAero file
    area : float, optional
        Referance area used to normalise coefficients, defaults to 0.0305128422 /m^2
    Attributes
    ----------
    area : float, optional
        Referance area used to normalise coefficients, /m^2
    COP : Scipy Interpolation Function
        Centre of pressure at time after ignition, when called interpolates to desired time /m
    CA : Scipy Interpolation Function
       Axial coefficient of drag, when called interpolates to desired time /
    CN : Scipy Interpolation Function
        Normal coefficient of drag, when called interpolates to desired time /
    
    """ 
    def __init__(self, file_location_string, area = 0.0305128422, error={"CA":1.0,"CN":1.0,"COP":1.0}): 
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
        CA = error["CA"]*np.array([CA_0, CA_2, CA_4])
        CN = error["CN"]*np.array([CN_0, CN_2, CN_4])
        COP = error["COP"]*0.0254*np.array([COP_0, COP_2, COP_4])    #Convert inches to m
        alpha = [0,2,4]
                    
        #Generate functions (note these are funcitons, not variables) which return a coefficient given (Mach, alpha)
        self.COP = scipy.interpolate.interp2d(Mach, alpha, COP)
        self.CA = scipy.interpolate.interp2d(Mach, alpha, CA)
        self.CN = scipy.interpolate.interp2d(Mach, alpha, CN)
                      
class Rocket:
    """The rocket and key simulation components
    Parameters
    ----------
    mass_model : Mass Model Object
        Mass model object, must have mass, ixx, iyy, izz, cog class methods which return them at time 
    motor : Motor Object
        Motor object that stores peformace parameters over time
    aero : Aero Object
        Most have area, cop, cn and ca class methods which return that at time
    launch_site : Launch site object
        Stores launch site parameters
    h : float, optional
        Timestep for integration (only required when variable is off), defaults ot 0.01 /s
    variable : bool, optional
        Adaptive timesteps?, defaults to False
    rtol : float
        Relative error tollerance for integration /
    atol : float
        Absolute error tollerance for integration /
    Attributes
    ----------
    mass_model : Mass Model Object
        Mass model object, must have mass, ixx, iyy, izz, cog class methods which return them at time 
    motor : Motor Object
        Motor object that stores peformace parameters over time
    aero : Aero Object
        Most have area, cop, cn and ca class methods which return that at time
    launch_site : Launch site object
        Stores launch site parameters
    h : float, optional
        Timestep for integration (only required when variable is off) /s
    variable : bool, optional
        Adaptive timesteps?, defaults to False
    rtol : float
        Relative error tollerance for integration /
    atol : float
        Absolute error tollerance for integration /
    b2i : Scipy Rotation Object
        Specifies the rotation from the body to inertial frame
    i2b : Scipy Rotation Object
        Specifies the rotation from the inertial to body frame
    pos_i : numpy array
        Position of the rocket in the inertial coordinate system [x,y,z] /m
    vel_i : numpy array
        Velocity of the rocket in the inertial coordinate system [x,y,z] /m/s
    w_b : numpy array
        Angular velocity of the body coordinate system in the inertial frame [x,y,z]/rad/s
    alt : float
        Altitude of the rocket (height abover surface in the launchsite frame) /m
    on_rail : bool
        Rocket still on rail? Initialises to True
    burn_out : bool
        Engine burned out? Initialises to False
    
    """   
    def __init__(self, mass_model, motor, aero, launch_site, h=0.01, variable=True, rtol=1e-7, atol=1e-14, parachute=Parachute(0,0,0,0,0,0),alt_poll_interval=1,thrust_vector=np.array([1,0,0]),errors={"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}):
        self.launch_site = launch_site
        self.motor = motor
        self.aero = aero
        self.mass_model = mass_model

        self.time = 0
        self.h = h

        self.variable_time = variable
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

        b2imat = np.zeros([3,3])
        b2imat[:,0] = xb_i
        b2imat[:,1] = yb_i
        b2imat[:,2] = zb_i
        self.b2i = Rotation.from_matrix(b2imat)     

        self.b2i = self.b2i                     #Body-to-Inertial Rotation - you can apply it to a vector with self.b2i.apply(vector)
        self.i2b = self.b2i.inv()               #Inertial-to-Body Rotation

        #Initialise angular positions and angular velocities
        self.pos_i = pos_l2i(np.array([0, 0, launch_site.alt]), launch_site, 0)                      #Position in inertial coordinates - defining the launch site origin as at an altitude of zero
        self.vel_i = vel_l2i([0,0,0], launch_site, 0)                                                #Velocity in intertial coordinates

        self.w_b = np.array([0,0,0])                                                 #Angular velocity in body coordinates
        self.alt = launch_site.alt                                                   #Altitude
        self.on_rail=True
        self.burn_out=False
        
        self.parachute_deployed=False
        self.parachute=parachute

        self.alt_record=self.altitude(self.pos_i,self.time)
        self.alt_poll_watch_interval=alt_poll_interval
        self.alt_poll_watch=self.alt_poll_watch_interval

        self.thrust_vector=thrust_vector
        self.env_vars = errors

    def aero_forces(self, pos_i, vel_i, b2i, w_b, time):  
        """Returns aerodynamic forces (in the body reference frame and the distance of the centre of pressure (COP) from the front of the vehicle.)
        Note
        ----
        -This currently ignores the damping moment generated by the rocket is rotating about its long axis
        -Unsure if the right angles for calculating CN as the angles of attack vary (same for CA as angles of attack vary)
        -Not sure if using the right density for converting between force coefficients and forces
        Parameters
        ----------
        pos_i : numpy array
            Position of the rocket in the inertial coordinate system [x,y,z] /m
        vel_i : numpy array
            Velocity of the rocket in the inertial coordinate system [x,y,z] /m/s
        b2i : scipy rotation object
            Defines the orientation of the body frame to the inertial frame
        w_b : numpy array
            Angular velocity of the body in the body frame [x,y,z] /rad/s
        time : float
            Time since ignition /s
        Returns
        -------
        numpy array
            Aerodynamic forces on the rocket in the body frame [x,y,z] /N
        float
            Distance from the front of the rocket that the forces act through /m
        """        
        #Use np.angle(ja + b) to replace np.arctan(a/b)
        alt = self.altitude(pos_i,time)
        
        #Get velocity relative to the wind (i.e. the airspeed vector), in body coordinates
        lat,long,alt=i2lla(pos_i,time)
        if alt<-5000:
            #I keep getting some weird error where if there is any wind the timesteps go to ~11s long near the ground and then it goes really far under ground, presumably in less than one whole timestep so the simulation can't break
            alt=-5000
        elif alt>81020:
            alt=81020
        v_rel_wind = b2i.inv().apply( direction_l2i((i2airspeed(pos_i, vel_i, self.launch_site, time) - self.launch_site.wind.get_wind(lat,long,alt)), self.launch_site, time) )
        v_a = np.linalg.norm(v_rel_wind)
        mach = v_a/(Atmosphere(alt).speed_of_sound[0]*self.env_vars["speed_of_sound"])

        #Angles of attack, as defined in Paper A: NASA Basic Considerations for Rocket Trajectory Simulation
        #np.angle(1j*a + b) is equivelant to np.arctan2(a/b) 
        #alpha = np.angle(1j*v_rel_wind[2] + v_rel_wind[0])
        beta = np.arctan2(v_rel_wind[1], (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.arctan2((v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5, v_rel_wind[0])
        alpha_star = np.arctan2(v_rel_wind[2], (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )
        #beta_star = np.angle(1j*v_rel_wind[1] + v_rel_wind[0])
        
        #Dynamic pressure at the current altitude and velocity - WARNING: Am I using the right density?
        q = 0.5*Atmosphere(alt).density[0]*self.env_vars["density"]*(v_a**2)
        
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
        return np.array([Fx,Fy,Fz]), COP, q, v_rel_wind
        
    def thrust(self, pos_i, vel_i, b2i, w_b, time, vector = [1,0,0]): 
        """Returns thrust and moments generated by the motor, in body frame.
        Note
        ----
        -Mainly derived from Joe Hunt's NOVUS Simulation
        Parameters
        ----------
        pos_i : numpy array
            Position of the rocket in the inertial coordinate system [x,y,z] /m
        vel_i : numpy array
            Velocity of the rocket in the inertial coordinate system [x,y,z] /m/s
        b2i : scipy rotation object
            Defines the orientation of the body frame to the inertial frame
        w_b : numpy array
            Angular velocity of the body in the body frame [x,y,z] /rad/s
        time : float
            Time since ignition /s
        vector : numpy array, optional
            Thrust direction in the body coordinate system - models misalignment or thrust vector control. Defaults to [1,0,0]
        Returns
        -------
        numpy array
            Thrust forces on the rocket in the body frame [x,y,z] /N
        """        
        vector = np.array(vector)
        alt = self.altitude(pos_i,time)

        if time < max(self.motor.motor_time_data):
            #Get the motor parameters at the current moment in time
            pres_cham = np.interp(time, self.motor.motor_time_data, self.motor.cham_pres_data)
            dia_throat = np.interp(time, self.motor.motor_time_data, self.motor.throat_data)
            gamma = np.interp(time, self.motor.motor_time_data, self.motor.gamma_data)
            nozzle_efficiency = np.interp(time, self.motor.motor_time_data, self.motor.nozzle_efficiency_data)
            pres_exit = np.interp(time, self.motor.motor_time_data, self.motor.exit_pres_data)
            nozzle_area_ratio = np.interp(time, self.motor.motor_time_data, self.motor.area_ratio_data)
            
            #Get atmospheric pressure (to calculate pressure thrust)
            pres_static = Atmosphere(alt).pressure[0]*self.env_vars["pressure"]
            
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
        
    def gravity(self, pos_i, time): 
        """Returns the gravity force, as a vector in inertial coordinates.
        Note
        ----
        -Uses a spherical Earth gravity model
        Parameters
        ----------
        pos_i : numpy array
            Position of the rocket in the inertial coordinate system [x,y,z] /m
        vel_i : numpy array
            Velocity of the rocket in the inertial coordinate system [x,y,z] /m/s
        b2i : scipy rotation object
            Defines the orientation of the body frame to the inertial frame
        w_b : numpy array
            Angular velocity of the body in the body frame [x,y,z] /rad/s
        time : float
            Time since ignition /s
        Returns
        -------
        numpy array
            Gravitational force on rocket in inertial frame [x,y,z] /N
        """   
        
        # F = -GMm/r^2 = μm/r^2 where μ = 3.986004418e14 for Earth
        return -self.env_vars["gravity"]*3.986004418e14 * self.mass_model.mass(time) * pos_i / np.linalg.norm(pos_i)**3

    def parachute_force(self, q, velocity, alt):
        c_d,s=self.parachute.get(alt)
        return -.5*q*s*c_d*velocity/np.linalg.norm(velocity)
    
    def altitude(self, pos_i, time):
        """Returns the altitude (height from surface in launch frame)
        Note
        ----
        -Uses a spherical Earth model
        Parameters
        ----------
        pos_i : numpy array
            Position of the rocket in the inertial coordinate system [x,y,z] /m
        Returns
        -------
        float
            Altitude /m
        """
        lat,longi,alt=i2lla(pos_i,time)
        return alt
    
    def accelerations(self, pos_i, vel_i, b2i, w_b, time):
        """Gathers the foces on the rocket and returns translational and rotational accelerations on the rocket
        Parameters
        ----------
        pos_i : numpy array
            Position of the rocket in the inertial coordinate system [x,y,z] /m
        vel_i : numpy array
            Velocity of the rocket in the inertial coordinate system [x,y,z] /m/s
        b2i : scipy rotation object
            Defines the orientation of the body frame to the inertial frame
        w_b : numpy array
            Angular velocity of the body in the body frame [x,y,z] /rad/s
        time : float
            Time since ignition /s
        Returns
        -------
        numpy array
            Translational accleration in inertial frame, and rotational acceleration using the body coordinate system
        """   
        if self.parachute_deployed==True and self.parachute.main_c_d!=0:
            aero_force_b, cop, q, v_rel_wind = self.aero_forces(pos_i, vel_i, b2i, w_b, time)
            cog = self.mass_model.cog(time)
            parachute_force_i=self.parachute_force(q,b2i.apply(v_rel_wind),self.altitude(pos_i,time))+self.gravity(pos_i,time)

            F=parachute_force_i
            Q_b=np.array([0,0,0])
        else:
            #Get all the forces in body coordinates
            thrust_b = self.thrust(pos_i, vel_i, b2i, w_b, time, self.thrust_vector)
            aero_force_b, cop, q, v_rel_wind = self.aero_forces(pos_i, vel_i, b2i, w_b, time)
            cog = self.mass_model.cog(time)
        
            #Get the moment arms
            r_engine_cog_b = (self.mass_model.l - cog)*np.array([-1,0,0])   #Vector (in body coordinates) of nozzle exit, relative to CoG
            r_cop_cog_b = (cop - cog)*np.array([-1,0,0])                    #Vector (in body coordinates) of CoP, relative to CoG
            
            #Calculate moments in body coordinates using moment = r x F    
            aero_moment_b = np.cross(r_cop_cog_b, aero_force_b)
            thrust_moment_b = np.cross(r_engine_cog_b, thrust_b)
            
            #Convert forces to inertial coordinates
            thrust_i = b2i.apply(thrust_b)
            aero_force_i = b2i.apply(aero_force_b)
            
            #Get total force and moment
            F = thrust_i + aero_force_i + self.gravity(pos_i,time)
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
        if self.on_rail==True:
            xb_i = b2i.apply([1,0,0])
            xb_i = xb_i/np.linalg.norm(xb_i)            #Normalise it just in case (but this step should be unnecessary)
            lin_acc = np.dot(lin_acc, xb_i)*xb_i        #Make it so we only keep the acceleration along the body's x-direction (i.e. in the forwards direction)
            wdot_b = np.array([0,0,0])                  #Assume no rotational acceleration on the rail

        #F = ma in inertial coordinates
        lin_acc = F/self.mass_model.mass(time)

        #If on the rail:
        if self.on_rail==True:
            xb_i = b2i.apply([1,0,0])
            xb_i = xb_i/np.linalg.norm(xb_i)            #Normalise it just in case (but this step should be unnecessary)
            lin_acc = np.dot(lin_acc, xb_i)*xb_i        #Make it so we only keep the acceleration along the body's x-direction (i.e. in the forwards direction)
            wdot_b = np.array([0,0,0])                  #Assume no rotational acceleration on the rail

        return np.stack([lin_acc, wdot_b])

    def fdot(self, time, fn):
        """Returns the rate of change of the Rocket's state array, f
        Notes
        -----
        -'fdot' here is the same as 'ydot' in the 2P1 (2nd Year) Engineering Lagrangian dynamics notes RK4 section
        Parameters
        ----------
        time : float
            Time since ignition /s
        fn : list
            [pos_i[0], pos_i[1], pos_i[2],
             vel_i[0], vel_i[1], vel_i[2],  
             w_b[0], w_b[1], w_b[2], 
             xb[0], xb[1], xb[2], 
             yb[0], yb[1], yb[2], 
             zb[0],zb[1],zb[2]]
        Returns
        -------
        numpy array
            [vel_i[0], vel_i[1], vel_i[2], 
            acc_i[0], acc_i[1], acc_i[2], 
            w_bdot[0], w_bdot[1], w_bdot[2], 
            xbdot[0], xbdot[1], xbdot[2], 
            ybdot[0], ybdot[1], ybdot[2], 
            zbdot[0], zbdot[1], zbdot[2]]
        """   

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
        acc_i, w_bdot = self.accelerations(pos_i, vel_i, b2i, w_b, time)

        #If a vector 'r' is rotating in the inertial frame, dr/dt = w_i x r
        xbdot = np.cross(w_i, xb)
        ybdot = np.cross(w_i, yb)
        zbdot = np.cross(w_i, zb)

        return np.array([vel_i[0],vel_i[1],vel_i[2], acc_i[0],acc_i[1],acc_i[2], w_bdot[0],w_bdot[1],w_bdot[2], xbdot[0],xbdot[1],xbdot[2], ybdot[0],ybdot[1],ybdot[2], zbdot[0],zbdot[1],zbdot[2]])

    def run(self, max_time=1000, debug=False, to_json = False):
        """Runs the rocket simulation
        Notes
        -----
        -Uses the scipy DOP853 O(h^8) integrator
        Parameters
        ----------
        max_time : int, optional
            Maximum simulation runtime, defaults to 300 /s
        debug : bool, optional
            Output more progress messages/warnings, defaults to False
        to_json : str, optional
            Export a .JSON file containing the data to the directory given, "False" means nothing will be exported.
        json_orient : string
            See pandas.DataFrame.to_json documentation - allowed values are: {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}. Defaults to 'split'.
        Returns
        -------
        pandas array
            Record of simulation, contains interial position and velocity, angular velocity in body coordinates, orientation and events (e.g. parachute).
            Most information can be derived from this in post processing.
            "time" : array
                List of times that all the data corresponds to /s
            "pos_i" : array
                List of inertial position vectors [x, y, z] /m
            "vel_i" : array
                List of inertial velocity vectors [x, y, z] /m/s
            "b2imat" : array:
                List of rotation matrices for going from the body to inertial coordinate system (i.e. a record of rocket orientation)
            "w_b" : array:
                List of angular velocity vectors, in body coordinates [x, y, z] /rad/s
            "events" : array:
                List of useful events        
        """
        if debug == True:
            print("Running simulation")

        xb_i = self.b2i.as_matrix()[:,0]
        yb_i = self.b2i.as_matrix()[:,1]
        zb_i = self.b2i.as_matrix()[:,2]

        fn = [self.pos_i[0],self.pos_i[1],self.pos_i[2],self.vel_i[0],self.vel_i[1],self.vel_i[2], self.w_b[0],self.w_b[1],self.w_b[2], xb_i[0],xb_i[1],xb_i[2],yb_i[0],yb_i[1],yb_i[2], zb_i[0],zb_i[1],zb_i[2]]
        integrator = integrate.DOP853(self.fdot,0,fn,1000,atol=self.atol,rtol=self.rtol)

        record=pd.DataFrame({})
        c=0
        while (self.altitude(self.pos_i,self.time)>=0 and self.time<max_time):
            if self.variable_time==False:
                integrator.h_abs=self.h
            events=self.check_phase(debug=debug)
            integrator.step()
            self.pos_i = np.array([integrator.y[0],integrator.y[1],integrator.y[2]])
            self.vel_i = np.array([integrator.y[3],integrator.y[4],integrator.y[5]])
            self.w_b = np.array([integrator.y[6],integrator.y[7],integrator.y[8]])
            b2imat = np.zeros([3,3])
            if self.parachute_deployed == False:
                b2imat[:,0] = np.array([integrator.y[9],integrator.y[10],integrator.y[11]])   #body x-direction
                b2imat[:,1] = np.array([integrator.y[12],integrator.y[13],integrator.y[14]])      #body y-direction
                b2imat[:,2] = np.array([integrator.y[15],integrator.y[16],integrator.y[17]])      #body z-direction
            else:
                lat,long,alt=i2lla(self.pos_i,self.time)
                wind_inertial =  vel_l2i(self.launch_site.wind.get_wind(lat,long,alt), self.launch_site, self.time)
                v_rel_wind = self.vel_i-wind_inertial
                b2imat[:,0] = -v_rel_wind/np.linalg.norm(v_rel_wind)   #body x-direction
                z=self.b2i.as_matrix()[:,2]
                b2imat[:,1] = np.cross(z,-v_rel_wind/np.linalg.norm(v_rel_wind))     #body y-direction
                b2imat[:,2] = z      #body z-direction
            

            self.b2i = Rotation.from_matrix(b2imat)
            self.i2b = self.b2i.inv()
                
            self.time = integrator.t
            if self.variable_time==True:
                self.h=integrator.h_previous

            new_row={"time":self.time,

                            "pos_i":self.pos_i.tolist(),
                            "vel_i":self.vel_i.tolist(),
                            "b2imat":b2imat.tolist(),
                            "w_b":self.w_b.tolist(),
                            
                            "events":events}

            record=record.append(new_row, ignore_index=True)
            if (c%100==0 and debug==True):
                print("t={:.2f} s alt={:.2f} km (h={} s). Step number {}".format(self.time, self.altitude(self.pos_i,self.time)/1000, integrator.h_abs, c))
            c+=1

        #Export a JSON if required
        if to_json == True:
            #Convert the DataFrame to a dict first, the in-built Python JSON library works better than panda's does I think
            dict = record.to_dict(orient="list")

            #Now use the inbuilt json module to export it
            with open(to_json, "w") as write_file:
                json.dump(dict, write_file)
            
            #How you could do this without the json module - but the JSON is stored in a less intuitive format:
            #record.to_json(path_or_buf = to_json, orient="split")

            if debug == True:
                print("Exported JSON data to '{}'".format(to_json))

        return record

    def check_phase(self, debug=False):
        """Checks phase of flight between steps
        Notes
        -----
        -Since this only checks between steps there may be a very short period where the rocket is still orientated as if its still on the rail when it is not
        -May look like the rocket leaves the rail at an altitude greater than the rail length for this reason
        Parameters
        ----------
        verbose : bool, optional
            Outputs progress messages if True
        Returns
        -------
        list
            List of events that happened in this step for log
        """
        events=[]
        if self.on_rail==True:
            flight_distance = np.linalg.norm(pos_i2l(self.pos_i,self.launch_site,self.time))
            if flight_distance>=self.launch_site.rail_length:
                if debug == True:
                    print("Cleared rail at t={:.2f} s with alt={:.2f} m and TtW={:.2f}".format(self.time,
                self.altitude(self.pos_i,self.time),
                np.linalg.norm(self.accelerations(self.pos_i, self.vel_i, self.b2i, self.w_b, self.time)[0])/9.81)
                )
                self.on_rail=False
                events.append("Cleared rail")

        if self.parachute_deployed==False:
            if (self.alt_poll_watch<self.time-self.alt_poll_watch_interval):
                current_alt=self.altitude(self.pos_i,self.time)
                if self.alt_record>current_alt:
                    if debug==True:
                        print("Parachute deployed at %sm at %ss"%(self.altitude(self.pos_i,self.time),self.time))
                    events.append("Parachute deployed")
                    self.parachute_deployed=True
                    self.w_b=np.array([0,0,0])
                    """wind_inertial =  vel_l2i(self.launch_site.wind, self.launch_site, self.time)
                    v_rel_wind = self.b2i.inv().apply(self.vel_i-wind_inertial)
                    xb_i = direction_l2i([0,0,1], self.launch_site, self.time)
                    yb_i = direction_l2i([0,1,0], self.launch_site, self.time)
                    zb_i = direction_l2i([-1,0,0], self.launch_site, self.time)
                    mat_b2i = np.zeros([3,3])
                    mat_b2i[:,0] = xb_i
                    mat_b2i[:,1] = yb_i
                    mat_b2i[:,2] = zb_i
                    self.b2i = Rotation.from_matrix(mat_b2i)  """ 
                else:
                    self.alt_poll_watch=self.time
                    self.alt_record=current_alt
        return events

def from_json(directory):
    """Imports simulation data from a JSON file
    Parameters
    ----------
    directory : string
        The directory of the simulation data .JSON file
    Returns
    -------
    pandas array
        Record of simulation, contains interial position and velocity, angular velocity in body coordinates, orientation and events (e.g. parachute).
        Most information can be derived from this in post processing.
        "time" : array
            List of times that all the data corresponds to /s
        "pos_i" : array
            List of inertial position vectors [x, y, z] /m
        "vel_i" : array
            List of inertial velocity vectors [x, y, z] /m/s
        "b2imat" : array:
            List of rotation matrices for going from the body to inertial coordinate system (i.e. a record of rocket orientation)
        "w_b" : array:
            List of angular velocity vectors, in body coordinates [x, y, z] /rad/s
        "events" : array:
            List of useful events        
    """
    #How you could try to do this without the json module:
    #return pd.read_json(directory, orient="split")

    #Import the JSON as a dict first (the in-built Python JSON library works better than panda's does I think)
    with open(directory, "r") as read_file:
        dict = json.load(read_file)
    
    #Now convert the dict to a pandas DataFrame
    return pd.DataFrame.from_dict(dict, orient="columns")