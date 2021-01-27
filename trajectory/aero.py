import scipy.interpolate, csv
import numpy as np

class AeroData:
    """Object holding aerodynamic data for the rocket.

    Assumes an axially symmetric body. Uses scipy.interpolate.interp2d to interpolate data from arrays.

    Args:
        CA_grid (array, 2D): Axial force coefficient data.
        CN_grid (array, 2D): Normal force coefficient data.
        COP_grid (array, 2D): Centre of pressure data, containing distances between the nose tip and the centre of pressure (m).
        Mach_list (array, 1D): Mach number data.
        alpha_list (array, 1D): Angle of attack data (radians).
        ref_area (float): Referance area used to normalise coefficients (m^2).
        pitch_damping_coefficient (float, optional): Pitch damping coefficient, defined by moment = C * ρ * ω^2.  Defaults to zero.
        roll_damping_coefficient (float, optional): Roll damping coefficient, defined by moment = C * ρ * ω^2. Defaults to zero.
        error(dictionary, optional): Used for running stochastic analyses.

    Attributes:
        CA_grid (array): Axial force coefficient data.
        CN_grid (array): Normal force coefficient data.
        COP_grid (array): Centre of pressure data, containing distances between the nose tip and the centre of pressure (m).
        Mach_list (array): Mach number data.
        alpha_list (array): Angle of attack data (radians).
        ref_area (float): Reference area used to normalise coefficients (m^2).
        pitch_damping_coefficient (float): Pitch damping coefficient, defined by moment = C * ρ * ω^2. 
        roll_damping_coefficient (float): Roll damping coefficient, defined by moment = C * ρ * ω^2. 
        error(dictionary): Used for running stochastic analyses.
    """

    def __init__(self, CA_grid, CN_grid, COP_grid, Mach_list, alpha_list, ref_area, pitch_damping_coefficient = 0.0, roll_damping_coefficient = 0.0, error={"CA":1.0,"CN":1.0,"COP":1.0}):
        self.ref_area = ref_area
        self.pitch_damping_coefficient = pitch_damping_coefficient
        self.roll_damping_coefficient = roll_damping_coefficient

        self.Mach_list = Mach_list
        self.alpha_list = alpha_list
        self.CA_grid = CA_grid
        self.CN_grid = CN_grid
        self.COP_grid = COP_grid

        self.error = error

        #These can be overridden to custom functions if you wanted - in which case the _grid attributes are irrelevant.
        self.CA_func = scipy.interpolate.interp2d(self.Mach_list, self.alpha_list, self.CA_grid)
        self.CN_func = scipy.interpolate.interp2d(self.Mach_list, self.alpha_list, self.CN_grid)
        self.COP_func = scipy.interpolate.interp2d(self.Mach_list, self.alpha_list, self.COP_grid)

    def CA(self, Mach, alpha):
        return self.CA_func(Mach, alpha)

    def CN(self, Mach, alpha):
        return self.CN_func(Mach, alpha)

    def COP(self, Mach, alpha):
        return self.COP_func(Mach, alpha)

    @staticmethod
    def from_lists(CA_list, CN_list, COP_list, Mach_list, alpha_list, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0, error={"CA":1.0,"CN":1.0,"COP":1.0}):
        #Convert into the right shapes
        Mach_list = np.unique(Mach_list)
        alpha_list = np.unique(alpha_list)

        CA_grid = np.reshape(CA_list, (len(alpha_list), len(Mach_list)))
        CN_grid = np.reshape(CN_list, (len(alpha_list), len(Mach_list)))
        COP_grid = np.reshape(COP_list, (len(alpha_list), len(Mach_list)))

        return AeroData(CA_grid, CN_grid, COP_grid, Mach_list, alpha_list, ref_area, pitch_damping_coefficient, roll_damping_coefficient, error)


    @staticmethod
    def from_rasaero(csv_directory, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0, error={"CA":1.0,"CN":1.0,"COP":1.0}): 
        """Convert an aerodynamic data .CSV file from RASAero II into an AeroData object.

        Args:
            csv_directory (string): Directory to .CSV file.
            ref_area (float): Referance area used to normalise coefficients (m^2).
            pitch_damping_coefficient (float, optional): Pitch damping coefficient, defined by moment = C * ρ * ω^2.  Defaults to zero.
            roll_damping_coefficient (float, optional): Roll damping coefficient, defined by moment = C * ρ * ω^2. Defaults to zero.
            error(dictionary, optional): Used for running stochastic analyses.

        Returns:
            AeroData: AeroData object.
        """
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

        #The data has length 7499 when it should be 7500 (3x2500).  We'll just add the last datapoint on twice.
        Mach_raw.append(Mach_raw[-1])
        alpha_raw.append(alpha_raw[-1])
        CA_raw.append(CA_raw[-1])
        COP_raw.append(COP_raw[-1])
        CN_raw.append(CN_raw[-1])

        #Convert alpha from degrees to radians
        alpha_raw = np.array(alpha_raw) * np.pi/180

        #Convert into the right forms
        Mach_list = np.unique(Mach_raw)
        alpha_list = np.unique(alpha_raw)

        CA_grid = np.reshape(CA_raw, (len(alpha_list), len(Mach_list)))
        CN_grid = np.reshape(CN_raw, (len(alpha_list), len(Mach_list)))
        COP_grid = np.reshape(COP_raw, (len(alpha_list), len(Mach_list)))

        return AeroData(CA_grid, CN_grid, COP_grid, Mach_list, alpha_list, ref_area, pitch_damping_coefficient, roll_damping_coefficient, error)

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