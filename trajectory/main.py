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
from .transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, i2lla, pos_i2alt
from ambiance import Atmosphere

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """A one line warning format"""
    return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line

def validate_lat_long(lat,long):
    """Makes latitude and longitude valid for wind"""
    if abs(lat)>90:
        lat=np.sign(lat)*(180-abs(lat))
        long+=180
    if long==-0.0:
        long=-0.0
    if long<0:
        long+=360
    long=np.mod(long,360)
    if lat==-0.0:
        lat=0.0
    return round(lat,4),round(long,4)

def closest(num,incriment):
    a=round(num/incriment)*incriment
    if a>num:
        b=a-.25
    else:
        b=a+.25
    return [a,b]

def points(lats,longs):
    points=[]
    for n in [0,1]:
        for m in [0,1]:
            points.append([lats[n],longs[m]])
    return points

class Wind:
    #Data will be strored in data_loc in the format lat_long_date_run_period.grb2 where lat and long are the bottom left values
    #Run has to be 00, 06, 12 or 18
    """Wind object
    Note
    ----
    Can give the wind vector for any lat long alt in the launch frame.
    Data collected from the NOAA's 0.25 degree 1 hour GFS forcast (https://nomads.ncep.noaa.gov/)

    Parameters
    ----------
    initial_lat : float
        Initial latitude /degrees
    initial_long : float
        Initial longitude /degrees
    data_loc : string, optional
        Route to folder where the data will be stored, defaults to data/wind/gfs
    variable : bool, optional
        Vary the wind or just use defaut for whole flight, defaults to True
    default : numpy array, optional
        Default wind vector [wind_x,wind_y,wind_z]/m/s, defauts to [0,0,0]
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000
    Attributes
    ----------
    centre_lat : float
        Initial latitude /degrees
    centre_long : float
        Initial longitude /degrees
    data_loc : string
        Route to folder where the data will be stored
    variable : bool
        Vary the wind or just use defaut for whole flight
    default : numpy array
        Default wind vector [wind_x,wind_y,wind_z]/m/s
    points : list
        List of available [latitude,longitude] points available
    date : string
        Date for forcast data in format YYYYMMDD
    forcast_time : string
        Forcast run time, must be 00,06,12 or 18
    run_time : string
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?)
    df : pandas DataFrame
        Dataframe holding wind data with columns lat, long, alt, wind x, wind y
    """
    def __init__(self,initial_long,initial_lat,variable=True,default=np.array([0,0,0]),data_loc="data/wind/gfs",run_date=date.today().strftime("%Y%m%d"),forcast_time="00",forcast_plus_time="000",fast=False):
        lat,long=validate_lat_long(initial_lat,initial_long)
        self.centre_lat=lat
        self.centre_long=long
        self.data_loc=data_loc#must be in last week for now
        self.variable = variable
        self.default=default
        self.points=[]
        self.fast=fast

        if lat<2:
            warnings.warn("Wind data robustness has not yet been tested for the equator")
        if abs(lat)>87:
            warnings.warn("Wind data robustness has not yet been tested near the poles")

        if variable == True:
            if forcast_time not in ["00","06","12","18"]:
                warnings.warn("The forcast run selected is not valid, must be '00', '06', '12' or '18'. This will be fatal on file load")
            valid_times=["00%s"%n for n in range(0,10)]+["0%s"%n for n in range(10,100)]+["%s"%n for n in range(100,385)]
            if forcast_plus_time not in valid_times:
                warnings.warn("The forcast time selected is not valid, must be three digit string time between 000 and 384. Thi siwll be fatal on file load")
            self.date=run_date
            self.forcast_time=forcast_time
            self.run_time=forcast_plus_time
            self.df,self.points=self.load_data(closest(self.centre_lat,.25),closest(self.centre_long,.25))
            
            if self.fast == True:
                self.winds=self.load_fast(lat,long)
    
    def load_fast(self,lat,long):
        mean=[]
        lats=closest(lat,.25)
        longs=closest(long,.25)
        x=[]
        y=[]
        for n in [0,1]:
            for m in [0,1]:
                x.append(scipy.interpolate.interp1d(self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["alt"],self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["w_x"], fill_value='extrapolate'))
                y.append(scipy.interpolate.interp1d(self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["alt"],self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["w_y"], fill_value='extrapolate'))
        
        for alt in np.linspace(0,45000,1000):
            mean.append(np.array([np.mean([x[0](alt),x[1](alt),x[2](alt),x[3](alt)]),np.mean([x[0](alt),x[1](alt),x[2](alt),x[3](alt)]),0]))
    
        return scipy.interpolate.interp1d(np.linspace(0,45000,1000),mean,fill_value='extrapolate')

    def load_data(self,lats,longs):
        """Loads wind data for particualr lat long to the objects df. 

        Notes
        -----
        Checks if the file corespondin to the requested lat long at the time and date of the object is available.
        If not downloads. Then reads into the dataframe.
        The file has cubes for geopotential height, wind x and wind y by pressure at a square grid of lat longs.
        The wind x and y are itterated throgh the pressures for each lat long and the altitude found for each point
        by finding the geopotential height at the particular pressure which can be converted to altitude.
        Each point is then stored in the df separatly for ease of searching (because the library Iris is complelty inept for this).

        Parameters
        ----------
        lat : float:
            Requested latitude /degrees
        longi : float:
            Requested longitude /degrees
        Returns
        -------
        pandas DataFrame
            The new wind points
        points
            list of [lat,long] not available
        """ 
        lat_top=max(lats)
        lat_bottom=min(lats)
        long_left=min(longs)
        long_right=max(longs)

        lat_top,long_left=validate_lat_long(lat_top,long_left)
        lat_bottom,long_right=validate_lat_long(lat_bottom,long_right)

        if not os.path.isfile("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time)):
            #This does download 3 rows that aren't needed but I can't work out how to yeet them
            print("Downloading files")
            if long_left>long_right:
                long_left_request=long_left-360
            else:
                long_left_request=long_left
            
            url="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t{run}z.pgrb2.0p25.f{hour}&lev_0.4_mb=on&lev_1000_mb=on&lev_100_mb=on&lev_10_mb=on&lev_150_mb=on&lev_15_mb=on&lev_180-0_mb_above_ground=on&lev_1_mb=on&lev_200_mb=on&lev_20_mb=on&lev_250_mb=on&lev_255-0_mb_above_ground=on&lev_2_mb=on&lev_300_mb=on&lev_30-0_mb_above_ground=on&lev_30_mb=on&lev_350_mb=on&lev_3_mb=on&lev_400_mb=on&lev_40_mb=on&lev_450_mb=on&lev_500_mb=on&lev_50_mb=on&lev_550_mb=on&lev_5_mb=on&lev_600_mb=on&lev_650_mb=on&lev_700_mb=on&lev_70_mb=on&lev_750_mb=on&lev_7_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&var_HGT=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon={leftlon}&rightlon={rightlon}&toplat={toplat}&bottomlat={bottomlat}&dir=%2Fgfs.{date}%2F{run}".format(leftlon=long_left_request,rightlon=long_right,toplat=lat_top,bottomlat=lat_bottom,date=self.date,run=self.forcast_time,hour=self.run_time)
            
            r = requests.get(url, stream=True)

            with open("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time),'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: 
                        f.write(chunk)
        if os.path.getsize("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time))<1000:
            raise RuntimeError("The weather data you requested was not found, this is usually because it was for an invalid date/time. lat=%s,long=%s was requested"%(lat,longi))
        data=iris.load("%s/%s_%s_%s_%s_%s.grb2"%(self.data_loc,lat_bottom,long_left,self.date,self.forcast_time,self.run_time))
        for index,row in enumerate(data):
            try:
                row.coord("pressure")
                if row.standard_name=="x_wind":
                    row_x_wind=index
                elif row.standard_name=="y_wind":
                    row_y_wind=index
                elif row.standard_name=="geopotential_height":
                    row_geo=index
            except:
                pass
        lats=list(data[row_geo].coord("latitude").points)
        longs=list(data[row_geo].coord("longitude").points)
        df=pd.DataFrame(columns=["lat","long","alt","w_x","w_y"])
        points=[]
        for long in longs:
            for lat in lats:
                press_1=data[row_x_wind].extract(iris.Constraint(latitude=lat,longitude=long)).coord("pressure").points
                press_2=data[row_y_wind].extract(iris.Constraint(latitude=lat,longitude=long)).coord("pressure").points
                press_3=data[row_geo].extract(iris.Constraint(latitude=lat,longitude=long)).coord("pressure").points
                press=[]
                for pres in press_1:
                    if (pres in press_2 and pres in press_3):
                        press.append(pres)
                for pres in press:
                    try:
                        if ([lat,long] not in self.points or [lat,long] not in points):
                            w_x=data[row_x_wind].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data
                            w_y=data[row_y_wind].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data
                            alt=10*metpy.calc.geopotential_to_height(data[row_geo].extract(iris.Constraint(latitude=lat,longitude=long,pressure=pres)).data*units.m**2/units.s**2).magnitude
                            row={"lat":lat,"long":np.mod(long,360),"alt":alt,"w_x":w_x,"w_y":w_y}
                            df=df.append(row,ignore_index=True)
                    except KeyError:
                        warnings.warn("Wind datapoint lat=%s, long=%s, pres=%s was missed because of an unknown Iris error, this is non fatal as it will be interpolated from other values"%(lat,longi,pres))
                    except:
                        warnings.warn("Wind datapoint lat=%s, long=%s, pres=%s was missed because of an unknown Iris error, this may be a fatal result if there are many instances in one dataset"%(lat,longi,pres))
                if len(df)!=0:
                    points.append([lat,long])
        
        return df,points

    def get_wind(self,lat,long,alt):
        """Returns wind for a specific lat,long,alt 
        
        Parameters
        ----------
        lat : float:
            Requested latitude /degrees
        longi : float:
            Requested longitude /degrees
        alt : float:
            Requested altitude /m
        Returns
        -------
        numpy array
            Wind speed vector [x,y,z]/m/s
        """ 
        lat,long=validate_lat_long(lat,long)
        if self.variable == True and self.fast==False:
            lats=closest(lat,.25)
            longs=closest(long,.25)
            availables=[[52.0,0.0],[52.25,0.25],[52.0,0.25],[52.25,0]]
            if not all(point in availables for point in points(lats,longs)):
                self.load_data(lats,longs)
            
            w=[[None,None],[None,None]]
            y=[None,None]
            for n in [0,1]:
                for m in [0,1]:
                    w[n][m]=scipy.interpolate.interp1d(self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["alt"],np.array([self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["w_x"],self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["w_y"],np.zeros(len(self.df.query("lat==%s"%lats[n]).query("long==%s"%longs[m])["w_y"]))]), fill_value='extrapolate')(alt)
                y[n]=w[n][0]+(long-longs[0])*(w[n][1]-w[n][0])/(longs[1]-longs[0])
                
            return y[0]+(lat-lats[0])*(y[1]-y[0])/(lats[1]-lats[0])
        elif self.variable == True and self.fast==True:
            return self.winds(alt)
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

        self.mdot_data = np.gradient(self.prop_mass_data, self.motor_time_data)       #Get the mass flow rates as an array, by doing d(prop_mass_data)/dt

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
    variable : bool, optional
        Vary the wind or just use defaut for whole flight, defaults to True
    default : numpy array, optional
        Default wind vector [wind_x,wind_y,wind_z]/m/s, defauts to [0,0,0]
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000
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
    def __init__(self, rail_length, rail_yaw, rail_pitch, alt, longi, lat, variable_wind=True,default_wind=np.array([0,0,0]),wind_data_loc="data/wind/gfs",run_date=date.today().strftime("%Y%m%d"),forcast_time="00",forcast_plus_time="000",fast_wind=False):
        self.rail_length = rail_length
        self.rail_yaw = rail_yaw
        self.rail_pitch = rail_pitch
        self.alt = alt+1e-5
        self.longi = longi
        self.lat = lat
        self.wind = Wind(longi,lat,variable=variable_wind,default=default_wind,data_loc=wind_data_loc,run_date=run_date,forcast_time=forcast_time,forcast_plus_time=forcast_plus_time,fast=fast_wind)
 
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
    errors : dictionary, optional
        Multiplied factor for the gravity, pressure, density and speed of sound used in the model,
        defaults to {"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}
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
    env_vars : dictionary
        Stores the coefficiants for the enviromental variables (see above)
    
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

        self.alt_record=pos_i2alt(self.pos_i,self.time)
        self.alt_poll_watch_interval=alt_poll_interval
        self.alt_poll_watch=self.alt_poll_watch_interval

        self.thrust_vector=thrust_vector
        self.env_vars = errors

    def aero_forces(self, pos_i, vel_i, b2i, w_b, time):  
        """Returns aerodynamic forces (in the body reference frame and the distance of the centre of pressure (COP) from the front of the vehicle.)
        Note
        ----
        -This currently ignores the damping moment generated by the rocket's rotation rate
        -Unsure if the right angles of attack for calculating CN and CA

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
        S = self.aero.ref_area
        
        #Drag/Force coefficients
        Cx = self.aero.CA(mach, abs(delta))[0]         #WARNING: Not sure if I'm using the right angles for these all
        Cz = self.aero.CN(mach, abs(alpha_star))[0]    #Or if this is the correct way to use CN
        Cy = self.aero.CN(mach, abs(beta))[0] 
        
        #Forces
        Fx = -np.sign(v_rel_wind[0])*Cx*q*S                         
        Fy = -np.sign(v_rel_wind[1])*Cy*q*S                         
        Fz = -np.sign(v_rel_wind[2])*Cz*q*S
        
        #Distance between rocket's nose tip and the COP:
        COP = self.aero.COP(mach, abs(delta))[0]
        
        #Return the forces (note that they're given using the body coordinate system, [x_b, y_b, z_b]).
        #Also return the distance that the COP is from the front of the rocket.
        return np.array([Fx,Fy,Fz]), COP, q, v_rel_wind
        
    def aero_damping_moment(self, pos_i, vel_i, b2i, w_b, time):  
        """Returns aerodynamic damping moments (in the body reference frame).

        Note
        ----
        -Assumes the damping coefficients are constants

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
            Aerodynamic damping moments on the rocket in the body frame [x,y,z] /N

        """        
        lat,long,alt=i2lla(pos_i,time)
        return np.array([-np.sign(w_b[0])*Atmosphere(alt).density[0] * w_b[0]**2 * self.aero.roll_damping_coefficient, 
                         -np.sign(w_b[1])*Atmosphere(alt).density[0] * w_b[1]**2 * self.aero.pitch_damping_coefficient,
                         -np.sign(w_b[2])*Atmosphere(alt).density[0] * w_b[2]**2 * self.aero.pitch_damping_coefficient])                  

    def thrust(self, pos_i, vel_i, b2i, w_b, time, vector = [1,0,0]): 
        """Returns thrust and moments generated by the motor, in body frame.
        Note
        ----
        - Mainly derived from Joe Hunt's NOVUS Simulation
        - Jet damping moment is calculated assuming the propellant COG is the same as that for the whole rocket. This will usually be less accurate if you have propellants near the top or bottom of the rocket.

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
        alt = pos_i2alt(pos_i,time)

        if time < max(self.motor.motor_time_data):
            #Get the motor parameters at the current moment in time
            pres_cham = np.interp(time, self.motor.motor_time_data, self.motor.cham_pres_data)
            dia_throat = np.interp(time, self.motor.motor_time_data, self.motor.throat_data)
            gamma = np.interp(time, self.motor.motor_time_data, self.motor.gamma_data)
            nozzle_efficiency = np.interp(time, self.motor.motor_time_data, self.motor.nozzle_efficiency_data)
            pres_exit = np.interp(time, self.motor.motor_time_data, self.motor.exit_pres_data)
            nozzle_area_ratio = np.interp(time, self.motor.motor_time_data, self.motor.area_ratio_data)
            mdot = np.interp(time, self.motor.motor_time_data, self.motor.mdot_data)

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
            mdot = 0
            if self.burn_out==False:
                print("Burnout at t={:.2f} s ".format(time))
            self.burn_out=True
        
        #Jet damping moment (due to the exhaust gases being rotated at w_b) - page 8 of https://apps.dtic.mil/sti/pdfs/AD0642855.pdf 
        #We will assume that the propellant COG is the same as the rocket COG.
        jet_damping_moment = mdot * (self.mass_model.cog(time) - self.mass_model.l)**2 * np.array([0, w_b[1], w_b[2]])

        #Return the thrust as a vector in body coordinates, and the jet damping moment
        return thrust*vector/np.linalg.norm(vector), jet_damping_moment
        
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
            parachute_force_i=self.parachute_force(q,b2i.apply(v_rel_wind),pos_i2alt(pos_i,time))+self.gravity(pos_i,time)

            F=parachute_force_i
            Q_b=np.array([0,0,0])
        else:
            #Get all the forces in body coordinates
            thrust_b, jet_damping_moment_b = self.thrust(pos_i, vel_i, b2i, w_b, time)
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
            Q_b = aero_moment_b + thrust_moment_b + jet_damping_moment_b + self.aero_damping_moment(pos_i, vel_i, b2i, w_b, time)
        
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
        while (pos_i2alt(self.pos_i,self.time)>=0 and self.time<max_time):
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
                print("t={:.2f} s alt={:.2f} km (h={} s). Step number {}".format(self.time, pos_i2alt(self.pos_i,self.time)/1000, integrator.h_abs, c))
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
                pos_i2alt(self.pos_i,self.time),
                np.linalg.norm(self.accelerations(self.pos_i, self.vel_i, self.b2i, self.w_b, self.time)[0])/9.81)
                )
                self.on_rail=False
                events.append("Cleared rail")

        if self.parachute_deployed==False:
            if (self.alt_poll_watch<self.time-self.alt_poll_watch_interval):
                current_alt=pos_i2alt(self.pos_i,self.time)
                if self.alt_record>current_alt:
                    if debug==True:
                        print("Parachute deployed at %sm at %ss"%(pos_i2alt(self.pos_i,self.time),self.time))
                    events.append("Parachute deployed")
                    self.parachute_deployed=True
                    self.w_b=np.array([0,0,0])
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