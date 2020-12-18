import ray,random,os,copy
import numpy as np
import pandas as pd
from .main import *
from .mass import CylindricalMassModel
from .transforms import pos_i2l, vel_i2l
from .aero import *

from .plot import *
from datetime import datetime
from datetime import date

def variable_name(**variables):
    return [x for x in variables][0]

def abs_stdev(value,percentage):
    return value*percentage

class StatisticalModel:
    """Stochastic model for the rocket flight

    Notes
    -----
    Every variable specified has a value and a standard deviation in a list (i.e. [mean,st_dev])
    Currenrly only cylindrical mass model is supported.
    Wind and pitch damping will currently not vary
    
    Parameters
    ----------
    launch_site_vars : dict
        Dictionary of variables for launch site object. Must contain: rail_length, rail_yaw, rail_pitch, alt, longi, lat
    mass_model_vars : dict
        Dictionary of variables for mass model object. Must contain: dry_mass, prop_mass, time_data, length, radius
    aero_file : string
        Location of RASAero data file
    aero_error : dict
        Standard deviaiton for the aero coefficients in format COP, CN and CA
    motor_base : MotorObject
        Unpeterbed motor object
    h : float, optional
        Default timestep /s, defaults to 0.05 - this doesn't really do anything
    variable_time : bool, optional
        Vary timesteps?, defaults to True
    alt_poll_interval : , optional
        Parachute altitude polling interval /s defaults to 1
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000
    thrust_error : float
        Standard deviation of thrust magnitude error /% 
    thrust_alignment : float
        Standard deviation of thrust alignment vector error 
    parachute_vars : dict
        Dictionary of variables for parachute object. Must contain: main_s, main_c_d, drogue_s, drogue_c_d, main_alt, attatch_distance
    env_vars : dictionary
        Multiplied factor for the gravity, pressure, density and speed of sound used in the model,
        defaults to {"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}

    Attributes
    ----------
    launch_site_vars : dict
        Dictionary of variables for launch site object. Must contain: rail_length, rail_yaw, rail_pitch, alt, longi, lat
    mass_model_vars : dict
        Dictionary of variables for mass model object. Must contain: dry_mass, prop_mass, time_data, length, radius
    aero_file : string
        Location of RASAero data file
    aero_error : dict
        Standard deviaiton for the aero coefficients in format COP, CN and CA
    motor_base : MotorObject
        Unpeterbed motor object
    h : float, optional
        Default timestep /s, defaults to 0.05 - this doesn't really do anything
    variable_time : bool, optional
        Vary timesteps?, defaults to True
    alt_poll_interval : , optional
        Parachute altitude polling interval /s defaults to 1
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000
    thrust_error : float
        Standard deviation of thrust magnitude error /% 
    thrust_alignment : float
        Standard deviation of thrust alignment vector error 
    parachute_vars : dict
        Dictionary of variables for parachute object. Must contain: main_s, main_c_d, drogue_s, drogue_c_d, main_alt, attatch_distance
    env_vars : dictionary
        Multiplied factor for the gravity, pressure, density and speed of sound used in the model,
        defaults to {"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}
    type_name : list
        The different types of errors to itterate over later
    wind_base : wind object
        Unpeterbed wind model
    """  
    def __init__(self, launch_site_vars, mass_model_vars, aero_file, aero_error, motor, thrust_error, thrust_alignment_error, parachute_vars, env_vars, h=0.05, variable=True,alt_poll_interval=1,run_date=date.today().strftime("%Y%m%d"),forcast_time="00",forcast_plus_time="000"):
        """Each variable set should be a list of [value,error]
        launch_site=[rail_length, rail_yaw, rail_pitch, alt, longi, lat, wind=[0,0,0]]
        """     
        self.launch_site_vars=launch_site_vars
        self.mass_model_vars=mass_model_vars
        self.aero_file=aero_file
        self.aero_vars=aero_error
        self.motor_base=motor
        self.h=h
        self.variable_time=variable
        self.thrust_error=thrust_error
        self.thrust_alignment_error=thrust_alignment_error
        self.parachute_vars=parachute_vars
        self.env_vars=env_vars
        self.type_names=["launch_site","mass_model","parachute","enviroment"]

        self.wind_base=model_wind=Wind(self.launch_site_vars["longi"][0],self.launch_site_vars["lat"][0],variable=True,run_date=run_date,forcast_time=forcast_time,forcast_plus_time=forcast_plus_time)

    @ray.remote
    def run_itteration(self, id, save_loc):
        """Runs an instance of the rocket with random errors

        Note
        ----
        Assumes gaussian errors for all variables, wind currenly not varied.

        Parameters
        ----------
        id : int
            Run number, used to name runs output
        save_loc : string
            Folder to store results
        """    
        run_vars={"launch_site":{k: np.random.normal(v[0],v[1]) for k, v in self.launch_site_vars.items()},#absolute errors given
                "mass_model":{k: np.array(v[0])*np.random.normal(1,v[1]) for k, v in self.mass_model_vars.items()},
                "parachute":{k: np.random.normal(v[0],abs_stdev(v[0],v[1])) for k, v in self.parachute_vars.items()},
                "aero":{k: np.array(v[0])*np.random.normal(1,v[1]) for k, v in self.aero_vars.items()},
                "env":{k: np.random.normal(1,v) for k,v in self.env_vars.items()}}
        run_vars["launch_site"]["alt"]=abs(run_vars["launch_site"]["alt"])#This doesn't work when mean alt is non zero but less than a few stdevs
        launch_site=LaunchSite(run_vars["launch_site"]["rail_length"],run_vars["launch_site"]["rail_yaw"],run_vars["launch_site"]["rail_pitch"],run_vars["launch_site"]["alt"],run_vars["launch_site"]["longi"],run_vars["launch_site"]["lat"],variable_wind=False)
        launch_site.wind=copy.copy(self.wind_base)
        #wind can now be modified for stats thing
        mass_model=CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        #mass_model=CylindricalMassModel(self.mass_model_vars["dry_mass"][0] + np.array(self.mass_model_vars["prop_mass"][0]),self.mass_model_vars["time_data"][0],self.mass_model_vars["length"][0],self.mass_model_vars["radius"][0])#CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        motor=copy.copy(self.motor_base)
        motor.nozzle_efficiency_data=np.array(motor.nozzle_efficiency_data)*np.random.normal(1,self.thrust_error)

        c_damp_pitch = pitch_damping_coefficient(run_vars["mass_model"]["length"], run_vars["mass_model"]["radius"], fin_number = run_vars["aero"]["fins"], area_per_fin = run_vars["aero"]["area_per_fin"])
        c_damp_roll = 0

        aero=RASAeroData(self.aero_file, run_vars["aero"]["ref_area"], c_damp_pitch, c_damp_roll, error={"COP":run_vars["aero"]["COP"],"CN":run_vars["aero"]["CN"],"CA":run_vars["aero"]["CA"]}) #I'm not convinces this is sufficient, should each datapoint not have its own random or is it okay to apply one error to the whole set?
        
        parachute=Parachute(run_vars["parachute"]["main_s"],run_vars["parachute"]["main_c_d"],run_vars["parachute"]["drogue_s"],run_vars["parachute"]["drogue_c_d"],run_vars["parachute"]["main_alt"],run_vars["parachute"]["attatch_distance"])
        
        #parachute=Parachute(self.parachute_vars["main_s"][0],self.parachute_vars["main_c_d"][0],self.parachute_vars["drogue_s"][0],self.parachute_vars["drogue_c_d"][0],self.parachute_vars["main_alt"][0],self.parachute_vars["attatch_distance"][0])
        
        thrust_alignment = np.array([np.random.normal(1,self.thrust_alignment_error),np.random.normal(0,self.thrust_alignment_error),np.random.normal(0,self.thrust_alignment_error)])
        thrust_alignment = thrust_alignment/np.linalg.norm(thrust_alignment)

        
        rocket=Rocket(mass_model, motor, aero, launch_site, h=self.h, variable=self.variable_time,parachute=parachute,thrust_vector=thrust_alignment,errors=run_vars["env"])#,errors=run_vars["enviroment"])
        run_output = rocket.run(debug=True)
        run_save = pd.DataFrame()
        run_save["time"]=run_output["time"]
        x,y,z=[],[],[]
        v_x,v_y,v_z=[],[],[]

        for index,pos in enumerate(run_output["pos_i"]):
            pos_l=pos_i2l(pos, rocket.launch_site, run_output["time"][index])
            x.append(pos_l[0])
            y.append(pos_l[1])
            z.append(pos_l[2])

            vel_l=vel_i2l(run_output["vel_i"][index], rocket.launch_site, run_output["time"][index])
            v_x.append(vel_l[0])
            v_y.append(vel_l[1])
            v_z.append(vel_l[2])

        run_save["x"]=x
        run_save["y"]=y
        run_save["z"]=z
        print(z[-1])
        run_save["v_x"]=v_x
        run_save["v_y"]=v_y
        run_save["v_z"]=v_z#These were'nt saving properly as vectors but really should

        with open("%s/%s.csv"%(save_loc,id), "w+") as f:
            run_save.to_csv(path_or_buf=f)
        
        self.wind_base=copy.copy(launch_site.wind)#incase more data has been downloaded

        

    def run_model(self,itters,save_loc=None):
        """Runs the stochastic model

        Parameters
        ----------
        itters : int
            Number of times to run the rocket
        save_loc : string, optional
            Folder to store results, defaults to None (is generated based on date if None)
        Returns
        -------
        string
            save location, if not specified is generated so needs to be returned to be known
        """    
        if save_loc==None:
            save_loc=os.path.join(os.getcwd(),"results/stat_model_%s"%datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        ray.init()
        for run in range(1,itters+1):
            self.run_itteration.remote(self,run,save_loc)
        input("Press enter when complete otherwise it pretends to have finished")
        return save_loc

def analyse(results_path, itterations, full_results=True, velocity=False):
    """Loads stats model results to put them in a more useful form for use, see stats_analysis_example notebook for example use

    Parameters
    ----------
    results_path : string
        Folder containing results
    itterations : int
        Number of runs used for the model
    full_results : bool, optional
        Return the full x,y,z,t for every run of the model
    velocity : bool, optional
        Return velocity analys, defaults to False - currently not implimented

    Returns
    -------
    if full_results=True
        numpy array
            landing position mean
        numpy array
            landing position covariant matrix
        numpy array 
            apogee position mean
        numpy array
            apogee position covariant matrix
        pandas dataframe
            positions of all apogees (columns x,y,z, row for each run)
        pandas dataframe
            positions of all landings (columns x,y,z, row for each run)
        pandas dataframe
            x position throughout flight (row for each run)
        pandas dataframe
            y position throughout flight (row for each run)
        pandas dataframe
            z position throughout flight (row for each run)
        andas dataframe
            time throughout flight (row for each run)
    else
        numpy array
            landing position mean
        numpy array
            landing position covariant matrix
        numpy array 
            apogee position mean
        numpy array
            apogee position covariant matrix
              
    """
    x,y,z,t=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    itts=range(1,itterations+1)

    for itt in itts:
        tmp=pd.read_csv("%s/%s.csv"%(results_path,itt))
    
        x=pd.concat([x,pd.DataFrame({itt:tmp["x"]})],ignore_index=True,axis=1)
        y=pd.concat([y,pd.DataFrame({itt:tmp["y"]})],ignore_index=True,axis=1)
        z=pd.concat([z,pd.DataFrame({itt:tmp["z"]})],ignore_index=True,axis=1)
        t=pd.concat([t,pd.DataFrame({itt:tmp["time"]})],ignore_index=True,axis=1)

    apogee=pd.DataFrame()
    apogee["index"]=z.idxmax()
    #I know this isn't the proper pandas way todo this but I can't see how todo it right
    apogee["x"]=[x[itt][apogee["index"].tolist()[itt]] for itt in range(0,itterations)]
    apogee["y"]=[y[itt][apogee["index"].tolist()[itt]] for itt in range(0,itterations)]
    apogee["alt"]=z.max()
    apogee=apogee.drop("index",axis=1)

    landing=pd.DataFrame()
    landing["x"]=[x[itt].dropna().iloc[-1] for itt in range(0,itterations)]
    landing["y"]=[y[itt].dropna().iloc[-1] for itt in range(0,itterations)]

    landing_mu=np.array([landing["x"].mean(),landing["y"].mean()])
    landing_cov=landing.cov()

    apogee_mu=np.array([apogee["x"].mean(),apogee["y"].mean(),apogee["alt"].mean()])
    apogee_cov=apogee.cov()

    if full_results==True:
        return landing_mu,landing_cov,apogee_mu,apogee_cov,apogee,landing,x,y,z,t
        #return x,y,z,t
    else:
        return landing_mu,landing_cov,apogee_mu,apogee_cov

