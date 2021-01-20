import numpy as np
import csv
import scipy.interpolate
import matplotlib.pyplot as plt

class Motor:
    """Object holding the performance data for the engine

    Notes
    ---------
    - Assumes constant nozzle exit area
    - Thrust must be given with an ambient pressure of 1 bar.

    Parameters
    ----------
    thrust : function(time)
        Function that returns the thrust (N), given the time since ignition (s).
    mdot : function(time)
        Function that returns the propellant mass flow rate (kg/s), given the time since ignition (s). This is used solely for jet damping calculations.
    exit_area : float
        Nozzle exit area (m^2)
    cut_off_time : float
        Time (s) since ignition at which the engine cuts off.
    ambient_pressure : float
        Ambient pressure corresponding to the thrust data (Pa). Defaults to 1e5 Pa (i.e. 1 bar).

    Attributes
    ----------
    thrust : function(time)
        Returns the thrust (N) given the time since ignition
    mdot : function(time)
        Returns the mass flow rate of propellant (kg/s), given the time since ignition. This is used purely for jet damping calculations.
    cut_off_time : float
        Time (s) since ignition at which the engine cuts off.
    exit_area : float
        Nozzle exit area (m^2)
    ambient_pressure : float
        Ambient pressure corresponding to the thrust data (Pa). 
    """   
      
    def __init__(self, thrust, mdot, cut_off_time, exit_area, ambient_pressure = 1e5):
        self.thrust = thrust
        self.mdot = mdot
        self.cut_off_time = cut_off_time
        self.exit_area = exit_area
        self.ambient_pressure = ambient_pressure

    @staticmethod
    def from_arrays(thrust_data, propellent_mass_data, time_data, exit_area, ambient_pressure = 1e5):
        mdot = np.gradient(propellent_mass_data, time_data)

        thrust_func = scipy.interpolate.interp1d(time_data, thrust_data)
        mdot_func = scipy.interpolate.interp1d(time_data, mdot)

        return Motor(thrust = thrust_func, 
                     mdot = mdot_func, 
                     cut_off_time = time[-1], 
                     exit_area = exit_area, 
                     ambient_pressure = ambient_pressure)
        
    @staticmethod
    def from_novus(csv_directory):
        '''Modified from Joe Hunt's NOVUS simulator'''

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
        mdot = np.gradient(prop_mass_data, time_data)
        
        #We want the thrust when the ambient pressure = 1 bar
        pres_ambient = 1e5
        
        #Calculate the thrust
        area_throat = ((dia_throat/2)**2)*np.pi
        exit_area = area_throat*nozzle_area_ratio
        thrust = (area_throat*pres_cham*(((2*gamma**2/(gamma-1))
                                            *((2/(gamma+1))**((gamma+1)/(gamma-1)))
                                            *(1-(pres_exit/pres_cham)**((gamma-1)/gamma)))**0.5)
                                            +(pres_exit-pres_ambient)*exit_area)

        thrust = thrust * nozzle_efficiency

        #Generate the functions and return the Motor object
        thrust_func = scipy.interpolate.interp1d(time_data, thrust)
        mdot_func = scipy.interpolate.interp1d(time_data, mdot)

        return Motor(thrust = thrust_func, 
                     mdot = mdot_func, 
                     cut_off_time = time_data[-1], 
                     exit_area = exit_area[0], 
                     ambient_pressure = pres_ambient)
