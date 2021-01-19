import numpy as np
import csv

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
    
    @staticmethod
    def from_novus(csv_directory):
        with open(csv_directory) as csvfile:
            motor_out = csv.reader(csvfile)

            (motor_time_data, prop_mass_data, cham_pres_data,
            throat_data, gamma_data, nozzle_efficiency_data,
            exit_pres_data, area_ratio_data) = [], [], [], [], [], [], [], []

            next(motor_out)
            for row in motor_out:
                motor_time_data.append(float(row[0]))
                prop_mass_data.append(float(row[1]))
                cham_pres_data.append(float(row[2]))
                throat_data.append(float(row[3]))
                gamma_data.append(float(row[4]))
                nozzle_efficiency_data.append(float(row[5]))
                exit_pres_data.append(float(row[6]))
                area_ratio_data.append(float(row[7]))

        return Motor(motor_time_data, 
                    prop_mass_data, 
                    cham_pres_data, 
                    throat_data, 
                    gamma_data, 
                    nozzle_efficiency_data, 
                    exit_pres_data, 
                    area_ratio_data)