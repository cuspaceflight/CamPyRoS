import ray,random,os,copy
import numpy as np
import pandas as pd
from .main import *
from .mass import CylindricalMassModel
from .transforms import pos_i2l, vel_i2l

from .plot import plot_ypr,plot_altitude_time
from datetime import datetime

def variable_name(**variables):
    return [x for x in variables][0]

def full_random():
    return 2*(random.random()-.5)


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
        run_vars={"launch_site":{},"mass_model":{},"parachute":{},"enviroment":{}}
        for index,item in enumerate([self.launch_site_vars,self.mass_model_vars,self.parachute_vars]):
            for key in item:
                run_vars[self.type_names[index]][key]=np.array(item[key][0])*(1+item[key][1]*full_random())

        run_vars["enviroment"]={k: 1+full_random()*v for k, v in self.env_vars.items()}

        launch_site=LaunchSite(run_vars["launch_site"]["rail_length"],run_vars["launch_site"]["rail_yaw"],run_vars["launch_site"]["rail_pitch"],run_vars["launch_site"]["alt"],run_vars["launch_site"]["longi"],run_vars["launch_site"]["lat"],run_vars["launch_site"]["wind"])
        mass_model=CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])#CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        motor=copy.copy(self.motor_base)
    
        motor.nozzle_efficiency_data=np.array(motor.nozzle_efficiency_data)*(1+self.thrust_error*full_random())
        aero_error = {k: full_random()*v for k, v in self.aero_error.items()}
        aero=RasAeroData(self.aero_file,variability=aero_error) #I'm not convinces this is sufficient, should each datapoint not have its own random or is it okay to apply one error to the whole set?
        parachute=Parachute(run_vars["parachute"]["main_s"],run_vars["parachute"]["main_c_d"],run_vars["parachute"]["drogue_s"],run_vars["parachute"]["drogue_c_d"],run_vars["parachute"]["main_alt"],run_vars["parachute"]["attatch_distance"])
        
        thrust_alignment = np.array([1,0,0])#+np.array([0,full_random()*self.thrust_alignment_error,full_random()*self.thrust_alignment_error])
        thrust_alignment = thrust_alignment/np.linalg.norm(thrust_alignment)

        
        rocket=Rocket(mass_model, motor, aero, launch_site, h=self.h, variable=self.variable_time,parachute=parachute,thrust_vector=thrust_alignment,errors=run_vars["enviroment"])
        run_output = rocket.run(max_time = 500)
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
        run_save["v_z"]=v_z
        #run_save["vel_l"] = [vel_i2l(pos, rocket.launch_site, run_output["time"][index]) for index,pos in enumerate(run_output["vel_i"])]
        with open("%s/%s.csv"%(save_loc,id), "w+") as f:
            run_save.to_csv(path_or_buf=f)

        

    def run_model(self,itters):
        save_loc=os.path.join(os.getcwd(),"results/stat_model_%s"%datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        ray.init()
        for run in range(1,itters+1):
            self.run_itteration.remote(self,run,save_loc)
        input("Press enter when complete otherwise it pretends to have finished")
        return save_loc