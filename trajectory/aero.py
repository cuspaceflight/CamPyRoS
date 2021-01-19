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
        Pitch damping coefficient, defined by m = C * ρ * ω^2.  Defaults to zero.
    roll_damping_coefficient : float
        Roll damping coefficient, defined by m = C * ρ * ω^2. Defaults to zero.
    COP : function or Scipy Interpolation Function
        Centre of pressure (distance from the nose tip) (m), as a function of Mach number and angle of attack. COP = COP(Mach, alpha).
    CA : function or Scipy Interpolation Function
       Axial coefficient of drag, as a function of Mach number and angle of attack. CA = CA(Mach, alpha).
    CN : function or Scipy Interpolation Function
        Normal coefficient of drag, as a function of Mach number and angle of attack. CN = CN(Mach, alpha).
    
    """ 
    def __init__(self, CA, CN, COP, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0):
        self.ref_area = ref_area
        self.pitch_damping_coefficient = pitch_damping_coefficient
        self.roll_damping_coefficient = roll_damping_coefficient

        self.CA = CA
        self.CN = CN
        self.COP = COP
        
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

class RASAeroData: 
    """Object holding aerodynamic data from a RASAero II 'Aero Plots' export file

    Note
    ----
    Relies on an axially symetric body

    Parameters
    ----------
    file_location_string : string
        Location of RASAero file
    area : float
        Referance area used to normalise coefficients, defaults to 0.0305128422 /m^2
    pitch_damping_coefficient : float, optional
        Pitch damping coefficient, defined by m = C * ρ * ω^2.  Defaults to zero.
    roll_damping_coefficient : float, optional
        Roll damping coefficient, defined by m = C * ρ * ω^2. Defaults to zero.

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
    def __init__(self, file_location_string, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0, error={"CA":1.0,"CN":1.0,"COP":1.0}): 
        print("Note: Using the RASAeroData Class is being phased out. Instead use AeroData.from_rasaero()")
        self.ref_area = ref_area
        self.pitch_damping_coefficient = pitch_damping_coefficient
        self.roll_damping_coefficient = roll_damping_coefficient

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