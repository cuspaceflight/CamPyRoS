import ray,random,os,copy
import numpy as np
import pandas as pd
from .main import *
from .mass import CylindricalMassModel
from .transforms import pos_i2l, vel_i2l

from .plot import *
from datetime import datetime

def variable_name(**variables):
    return [x for x in variables][0]

def abs_stdev(value,percentage):
    return value*percentage

class StatisticalModel:
    """Class for monte carlo modeling of flights
    """    
    def __init__(self, launch_site_vars, mass_model_vars, aero_file, aero_error, motor, thrust_error, thrust_alignment_error, parachute_vars, env_vars, h=0.05, variable=True,alt_poll_interval=1):
        """Each variable set should be a list of [value,error]
        launch_site=[rail_length, rail_yaw, rail_pitch, alt, longi, lat, wind=[0,0,0]]
        """     
        self.launch_site_vars=launch_site_vars
        self.mass_model_vars=mass_model_vars
        self.aero_file=aero_file
        self.aero_error=aero_error
        self.motor_base=motor
        self.h=h
        self.variable_time=variable
        self.thrust_error=thrust_error
        self.thrust_alignment_error=thrust_alignment_error
        self.parachute_vars=parachute_vars
        self.env_vars=env_vars
        self.type_names=["launch_site","mass_model","parachute","enviroment"]

    @ray.remote
    def run_itteration(self, id, save_loc):
        run_vars={"launch_site":{k: np.random.normal(v[0],v[1]) for k, v in self.launch_site_vars.items()},#absolute errors given
                "mass_model":{k: np.array(v[0])*np.random.normal(1,v[1]) for k, v in self.mass_model_vars.items()},
                "parachute":{k: np.random.normal(v[0],abs_stdev(v[0],v[1])) for k, v in self.parachute_vars.items()},
                "aero":{k: np.random.normal(1,v) for k,v in self.aero_error.items()},
                "env":{k: np.random.normal(1,v) for k,v in self.env_vars.items()}}
        run_vars["launch_site"]["alt"]=abs(run_vars["launch_site"]["alt"])#This doesn't work when mean alt is non zero but less than a few stdevs
        launch_site=LaunchSite(run_vars["launch_site"]["rail_length"],run_vars["launch_site"]["rail_yaw"],run_vars["launch_site"]["rail_pitch"],run_vars["launch_site"]["alt"],run_vars["launch_site"]["longi"],run_vars["launch_site"]["lat"],run_vars["launch_site"]["wind"])
        mass_model=CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        #mass_model=CylindricalMassModel(self.mass_model_vars["dry_mass"][0] + np.array(self.mass_model_vars["prop_mass"][0]),self.mass_model_vars["time_data"][0],self.mass_model_vars["length"][0],self.mass_model_vars["radius"][0])#CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        motor=copy.copy(self.motor_base)
        motor.nozzle_efficiency_data=np.array(motor.nozzle_efficiency_data)*np.random.normal(1,self.thrust_error)

        aero=RasAeroData(self.aero_file,error=run_vars["aero"]) #I'm not convinces this is sufficient, should each datapoint not have its own random or is it okay to apply one error to the whole set?
        
        parachute=Parachute(run_vars["parachute"]["main_s"],run_vars["parachute"]["main_c_d"],run_vars["parachute"]["drogue_s"],run_vars["parachute"]["drogue_c_d"],run_vars["parachute"]["main_alt"],run_vars["parachute"]["attatch_distance"])
        
        #parachute=Parachute(self.parachute_vars["main_s"][0],self.parachute_vars["main_c_d"][0],self.parachute_vars["drogue_s"][0],self.parachute_vars["drogue_c_d"][0],self.parachute_vars["main_alt"][0],self.parachute_vars["attatch_distance"][0])
        
        thrust_alignment = np.array([np.random.normal(1,self.thrust_alignment_error),np.random.normal(0,self.thrust_alignment_error),np.random.normal(0,self.thrust_alignment_error)])
        thrust_alignment = thrust_alignment/np.linalg.norm(thrust_alignment)

        
        rocket=Rocket(mass_model, motor, aero, launch_site, h=self.h, variable=self.variable_time,parachute=parachute,thrust_vector=thrust_alignment,errors=run_vars["env"])#,errors=run_vars["enviroment"])
        run_output = rocket.run()
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
        run_save["v_x"]=v_x
        run_save["v_y"]=v_y
        run_save["v_z"]=v_z#These were'nt saving properly as vectors but really should

        with open("%s/%s.csv"%(save_loc,id), "w+") as f:
            run_save.to_csv(path_or_buf=f)

        

    def run_model(self,itters,save_loc=None):
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
    x,y,z=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    itts=range(1,itterations+1)

    for itt in itts:
        tmp=pd.read_csv("%s/%s.csv"%(results_path,itt))
    
        x[itt]=tmp["x"]
        y[itt]=tmp["y"]
        z[itt]=tmp["z"]

    apogee=pd.DataFrame()
    apogee["index"]=z.idxmax()
    apogee["alt"]=z.max()
    #I know this isn't the proper pandas way todo this but I can't see how todo it right
    apogee["x"]=[x[itt][apogee["index"][itt]] for itt in itts]
    apogee["y"]=[y[itt][apogee["index"][itt]] for itt in itts]
    apogee=apogee.drop("index",axis=1)

    landing=pd.DataFrame()
    landing["x"]=[x[itt].dropna().iloc[-1] for itt in itts]
    landing["y"]=[y[itt].dropna().iloc[-1] for itt in itts]

    landing_mu=np.array([landing["x"].mean(),landing["y"].mean()])
    landing_cov=landing.cov()

    apogee_mu=np.array([apogee["x"].mean(),apogee["y"].mean(),apogee["alt"].mean()])
    apogee_cov=apogee.cov()

    if full_results==True:
        return landing_mu,landing_cov,apogee_mu,apogee_cov,apogee,landing
    else:
        return landing_mu,landing_cov,apogee_mu,apogee_cov

