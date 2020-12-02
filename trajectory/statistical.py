import ray,random,os,copy
import numpy as np
from .main import *
from .mass import CylindricalMassModel

from .plot import plot_ypr,plot_altitude_time
from datetime import datetime

def variable_name(**variables):
    return [x for x in variables][0]

def full_random():
    return 2*(random.random()-.5)


class StatisticalModel:
    """Class for monte carlo modeling of flights
    """    
    def __init__(self, launch_site_vars, mass_model_vars, aero_file, aero_error, motor, thrust_error, thrust_alignment_error, parachute_vars, h=0.05, variable=True,alt_poll_interval=1):
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
        self.type_names=["launch_site","mass_model","parachute"]

    #@ray.remote
    def run_itteration(self, id, save_loc):
        run_vars={"launch_site":{},"mass_model":{},"parachute":{}}
        for index,item in enumerate([self.launch_site_vars,self.mass_model_vars,self.parachute_vars]):
            for key in item:
                run_vars[self.type_names[index]][key]=np.array(item[key][0])#*(1+item[key][1]*full_random())

        launch_site=LaunchSite(self.launch_site_vars["rail_length"][0],self.launch_site_vars["rail_yaw"][0],self.launch_site_vars["rail_pitch"][0],self.launch_site_vars["alt"][0],self.launch_site_vars["longi"][0],self.launch_site_vars["lat"][0],self.launch_site_vars["wind"][0])
        #mass_model=CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])#CylindricalMassModel(run_vars["mass_model"]["dry_mass"] + run_vars["mass_model"]["prop_mass"],run_vars["mass_model"]["time_data"],run_vars["mass_model"]["length"],run_vars["mass_model"]["radius"])
        #mass_model=CylindricalMassModel(self.mass_model_vars["dry_mass"][0]*(1+self.mass_model_vars["dry_mass"][1]*full_random()) + np.array(self.mass_model_vars["prop_mass"][0])*(1+self.mass_model_vars["prop_mass"][1]*full_random()), np.array(self.mass_model_vars["time_data"][0]), self.mass_model_vars["length"][0], self.mass_model_vars["radius"][0])

        mass_model=CylindricalMassModel(self.mass_model_vars["dry_mass"][0] + np.array(self.mass_model_vars["prop_mass"][0]), np.array(self.mass_model_vars["time_data"][0]), self.mass_model_vars["length"][0], self.mass_model_vars["radius"][0])
        motor=copy.copy(self.motor_base)
        #motor.nozzle_efficiency_data=[(1+self.thrust_error*full_random())*point for point in motor.nozzle_efficiency_data]#This is a hack since nozzle efficiency is directly proportional to thrust
        aero_error = {k: full_random()*v for k, v in self.aero_error.items()}
        aero=RasAeroData(self.aero_file)#,variability=aero_error) #I'm not convinces this is sufficient, should each datapoint not have its own random or is it okay to apply one error to the whole set?
        #parachute=Parachute(run_vars["parachute"]["main_s"],run_vars["parachute"]["main_c_d"],run_vars["parachute"]["drogue_s"],run_vars["parachute"]["drogue_c_d"],run_vars["parachute"]["main_alt"],run_vars["parachute"]["attatch_distance"])
        parachute=Parachute(self.parachute_vars["main_s"][0]*(1+full_random()*self.parachute_vars["main_s"][1]),self.parachute_vars["main_c_d"][0]*(1+full_random()*self.parachute_vars["main_c_d"][1]),self.parachute_vars["drogue_s"][0]*(1+full_random()*self.parachute_vars["drogue_s"][1]),self.parachute_vars["drogue_c_d"][0]*(1+full_random()*self.parachute_vars["drogue_c_d"][1]),self.parachute_vars["main_alt"][0]*(1+full_random()*self.parachute_vars["main_alt"][1]),self.parachute_vars["attatch_distance"][0]*(1+full_random()*self.parachute_vars["attatch_distance"][1]))
        thrust_alignment = np.array([1,0,0])+np.array([0,full_random()*self.thrust_alignment_error,full_random()*self.thrust_alignment_error])
        thrust_alignment = thrust_alignment/np.linalg.norm(thrust_alignment)
        print(thrust_alignment)
        rocket=Rocket(mass_model, motor, aero, launch_site, h=self.h, variable=self.variable_time,parachute=parachute,thrust_vector=thrust_alignment)
        run_output = rocket.run(max_time = 300, debug=True)

        plot_ypr(run_output,rocket)
        plot_altitude_time(run_output,rocket)

        with open("%s/%s.csv"%(save_loc,id), "w+") as f:
            run_output.to_csv(path_or_buf=f)

        

    def run_model(self,itters):
        save_loc=os.path.join(os.getcwd(),"results/stat_model_%s"%datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        ray.init()
        for run in range(1,itters+1):
            self.run_itteration.remote(self,run,save_loc)
        input("Press enter when complete otherwise it pretends to have finished")
        return save_loc