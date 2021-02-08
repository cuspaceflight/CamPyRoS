import numpy as np
import csv
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd

__copyright__ = """

    Copyright 2021 Jago Strong-Wright & Daniel Gibbons

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"

class Motor:
    """Object for holding rocket engine data. 
    
    Assumes constant nozzle exit area.

    Args:
        thrust_array (list): Thrust data (N).
        time_array (list): Times corresponding to thrust_array data points (s).
        exit_area (float): Nozzle exit area (m^2).
        pos (float): Distance between the nose tip and the point at which the thrust acts (m).
        ambient_pressure (float, optional): Ambient pressure used to obtain the thrust_array data (Pa). Defaults to 1e5.

    Attributes:
        thrust_array (list): Thrust data (N).
        time_array (list): Times corresponding to thrust_array data points (s).
        exit_area (float): Nozzle exit area (m^2).
        pos (float): Distance between the nose tip and the point at which the thrust acts (m).
        ambient_pressure (float, optional): Ambient pressure used to obtain the thrust_array data (Pa).

    """
      
    def __init__(self, thrust_array, time_array, exit_area, pos, ambient_pressure = 1e5):

        self.thrust_array = thrust_array            #Thrust data (N)
        self.time_array = time_array                #Times corresponding to thrust_array data points (s)
        self.pos = pos                              #Distance between the nose tip and the point at which the thrust acts (m)
        self.exit_area = exit_area                  #Nozzle exit area (m^2)
        self.ambient_pressure = ambient_pressure    #Ambient pressure used to obtain the thrust_array data (Pa)

    def thrust(self, time):
        """Function for calculating the thrust at a given time, with an ambient pressure of self.ambient_pressure.

        Args:
            time (float): Time since ignition (s).

        Returns:
            float: Thrust (N).
        """
        return np.interp(time, self.time_array, self.thrust_array)

    @staticmethod
    def from_novus(csv_directory, pos):
        """Generate Motor object from a novus_sim_6 output csv file. Modified from Joe Hunt's NOVUS simulator.

        Args:
            csv_directory (string): Directory of the .CSV file.
            pos (float): Distance between the nose tip and the point at which the thrust acts (m).

        Returns:
            Motor: The Motor object.
        """

        #Collect data from the CSV
        with open(csv_directory) as csvfile:
            motor_out = csv.reader(csvfile)

            (time_data, prop_mass_data, cham_pres_data,
            throat_data, gamma_data, nozzle_efficiency_data,
            exit_pres_data, area_ratio_data) = [], [], [], [], [], [], [], []

            next(motor_out)
            for row in motor_out:
                time_data.append(float(row[0]))
                prop_mass_data.append(float(row[1]))
                cham_pres_data.append(float(row[2]))
                throat_data.append(float(row[3]))
                gamma_data.append(float(row[4]))
                nozzle_efficiency_data.append(float(row[5]))
                exit_pres_data.append(float(row[6]))
                area_ratio_data.append(float(row[7]))

        #Convert everything into numpy arrays so we can do element-wise operations
        pres_cham = np.array(cham_pres_data)
        dia_throat = np.array(throat_data)
        gamma = np.array(gamma_data)
        nozzle_efficiency = np.array(nozzle_efficiency_data)
        pres_exit = np.array(exit_pres_data)
        nozzle_area_ratio = np.array(area_ratio_data)
        
        #Let's use ambient pressure = 1 bar
        pres_ambient = 1e5
        
        #Calculate the thrust
        area_throat = ((dia_throat/2)**2)*np.pi
        exit_area = area_throat*nozzle_area_ratio
        thrust = (area_throat*pres_cham*(((2*gamma**2/(gamma-1))
                                            *((2/(gamma+1))**((gamma+1)/(gamma-1)))
                                            *(1-(pres_exit/pres_cham)**((gamma-1)/gamma)))**0.5)
                                            +(pres_exit-pres_ambient)*exit_area)

        thrust = thrust * nozzle_efficiency

        return Motor(thrust_array = thrust, 
                     time_array = time_data,
                     exit_area = exit_area[0], 
                     pos = pos,
                     ambient_pressure = pres_ambient)

def load_motor(file):
    """Legacy requirment for statistical models

    Args:
        file (string): Location of novus output file

    Returns:
        dict: Motor data - "motor_time","prop_mass",
            "cham_pres","throat","gamma", "nozzle_efficiency",
            "exit_pres","area_ratio","vmass","lden","lmass",
            "fuel_mass","density_fuel","dia_fuel","length_port"
    """  
    motor_csv = pd.read_csv(file)

    time_array = motor_csv['Time']
    smass_array = motor_csv['Solid Fuel Mass (kg)']
    S_DEN = motor_csv['Solid Fuel Density (kg/m^3)'][0]
    S_L = motor_csv['Solid Fuel Length (m)'][0]
    S_ROUT = motor_csv['Solid Fuel Outer Diameter (m)'][0]
    vmass_array =  motor_csv['Vapour Mass (kg)']
    vden_array = motor_csv['Vapour Density (kg/m^3)']
    lmass_array = motor_csv['Liquid Mass (kg)']
    lden_array = motor_csv['Liquid Density (kg/m^3)']
        
    return {"time":time_array,
            "smass":smass_array,
            "sden":S_DEN,
            "s_l":S_L,
            "s_rout":S_ROUT,
            "vmass":vmass_array,
            "vden":vden_array,
            "lmass":lmass_array,
            "lden":lden_array}