import scipy.interpolate, csv
import numpy as np

class AeroData:
    """Object holding aerodynamic data for the rocket

    Note
    ----
    Relies on an axially symetric body

    Attributes
    ----------
    ref_area : float, optional
        Referance area used to normalise coefficients, /m^2
    pitch_damping_coefficient : float
        Pitch damping coefficient, defined by moment = C * ρ * ω^2.  Defaults to zero.
    roll_damping_coefficient : float
        Roll damping coefficient, defined by moment = C * ρ * ω^2. Defaults to zero.
    COP : function(Mach, alpha)
        Distance between the nose tip and the centre of pressure (m), as a function of Mach number and angle of attack.
    CA : function(Mach, alpha)
       Axial coefficient of drag, as a function of Mach number and angle of attack.
    CN : function(Mach, alpha)
        Normal coefficient of drag, as a function of Mach number and angle of attack. 
    
    """ 
    def __init__(self, CA, CN, COP, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0):
        self.ref_area = ref_area
        self.pitch_damping_coefficient = pitch_damping_coefficient
        self.roll_damping_coefficient = roll_damping_coefficient

        self.CA = CA
        self.CN = CN
        self.COP = COP

    @staticmethod
    def from_arrays(CA_data, CN_data, COP_data, Mach_data, alpha_data, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0):
        #Generate the functions from the arrays, using scipy.interpolate.interp2d
        COP_function = scipy.interpolate.interp2d(Mach_data, alpha_data, COP_data)
        CA_function = scipy.interpolate.interp2d(Mach_data, alpha_data, CA_data)
        CN_function = scipy.interpolate.interp2d(Mach_data, alpha_data, CN_data)

        return AeroData(CA_function, CN_function, COP_function, ref_area, pitch_damping_coefficient, roll_damping_coefficient)

    @staticmethod
    def from_rasaero(csv_directory, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0, error={"CA":1.0,"CN":1.0,"COP":1.0}): 
        with open(csv_directory) as csvfile:
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
        COP_function = scipy.interpolate.interp2d(Mach, alpha, COP)
        CA_function = scipy.interpolate.interp2d(Mach, alpha, CA)
        CN_function = scipy.interpolate.interp2d(Mach, alpha, CN)

        return AeroData(CA_function, CN_function, COP_function, ref_area, pitch_damping_coefficient, roll_damping_coefficient)

def pitch_damping_coefficient(length, radius, fin_number, area_per_fin):
    """Gives approximate values for the pitch damping coefficient. Uses equations (3.59) and (3.60) from the OpenRocket documentation.

    Note
    ----
    In this model we define the pitch damping coefficient as:
        m = C * ρ * ω^2
    Where:
        m = moment 
        ρ = free-stream density 
        ω = pitch rate
        C = pitch damping coefficient.

    Assumptions:
    - Fins are at the very bottom of the rocket
    - COG of the rocket is half way up the length  

    Parameters
    ----------
    length : float
        Length of the rocket (m)
    radius : float
        Radius of the rocket (assuming it's a cylinder) (m)
    fin_number: int
        Number of fins on the rocket
    area_per_fin : float
        Area of a single fin (m^2)

    Returns
    ----------
    Pitch damping cofficient
    """ 

    if fin_number > 4:
        fin_number = 4

    return 0.275*radius*(length**4) + 0.3*fin_number*area_per_fin*(length/2)**3