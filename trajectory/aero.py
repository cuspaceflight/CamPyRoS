import scipy.interpolate, csv
import matplotlib.pyplot as plt
import numpy as np

class AeroData:
    """Object holding aerodynamic data for the rocket.

    Assumes an axially symmetric body. Uses scipy.interpolate.interp2d to interpolate data from arrays.

    Args:
        CA_grid (array, 2D): Axial force coefficient data.
        CN_grid (array, 2D): Normal force coefficient data.
        COP_grid (array, 2D): Centre of pressure data (m), containing distances between the nose tip and the centre of pressure.
        Mach_grid (array, 1D or 2D): Mach number data.
        alpha_grid (array, 1D or 2D): Angle of attack data (radians).
        ref_area (float): Referance area used to normalise coefficients (m^2).
        pitch_damping_coefficient (float, optional): Pitch damping coefficient, defined by moment = C * ρ * ω^2.  Defaults to zero.
        roll_damping_coefficient (float, optional): Roll damping coefficient, defined by moment = C * ρ * ω^2. Defaults to zero.
        error(dictionary, optional): Used for running stochastic analyses. Defaults to {"CA":1.0,"CN":1.0,"COP":1.0}.

    Attributes:
        CA_grid (array): Axial force coefficient data.
        CN_grid (array): Normal force coefficient data.
        COP_grid (array): Centre of pressure data, containing distances between the nose tip and the centre of pressure (m).
        Mach_grid (array): Mach number data.
        alpha_grid (array): Angle of attack data (rad).
        ref_area (float): Reference area used to normalise coefficients (m^2).
        pitch_damping_coefficient (float): Pitch damping coefficient, defined by moment = C * ρ * ω^2. 
        roll_damping_coefficient (float): Roll damping coefficient, defined by moment = C * ρ * ω^2. 
        error(dictionary): Used for running stochastic analyses.
    """

    def __init__(self, CA_grid, CN_grid, COP_grid, Mach_grid, alpha_grid, ref_area, pitch_damping_coefficient = 0.0, roll_damping_coefficient = 0.0, error={"CA":1.0,"CN":1.0,"COP":1.0}):
        self.ref_area = ref_area
        self.pitch_damping_coefficient = pitch_damping_coefficient
        self.roll_damping_coefficient = roll_damping_coefficient

        self.Mach_grid = Mach_grid
        self.alpha_grid = alpha_grid
        self.CA_grid = CA_grid
        self.CN_grid = CN_grid
        self.COP_grid = COP_grid

        self.error = error

        #These can be overridden to custom functions if you wanted - in which case the _grid attributes are irrelevant.
        self.CA_func = scipy.interpolate.interp2d(self.Mach_grid, self.alpha_grid, self.CA_grid)
        self.CN_func = scipy.interpolate.interp2d(self.Mach_grid, self.alpha_grid, self.CN_grid)
        self.COP_func = scipy.interpolate.interp2d(self.Mach_grid, self.alpha_grid, self.COP_grid)

    def CA(self, Mach, alpha):
        return self.error["CA"] * self.CA_func(Mach, alpha)

    def CN(self, Mach, alpha):
        return self.error["CN"] * self.CN_func(Mach, alpha)

    def COP(self, Mach, alpha):
        return self.error["COP"] * self.COP_func(Mach, alpha)
    
    def show_plot(self, Mach = np.linspace(0, 25, 500), alpha = np.linspace(0, 4, 5)*np.pi/180):
        """"Shows plots of the CA, CN and COP functions, so you can visually check if the system has interpreted your data correctly.

        Args:
            Mach (array): Array of Mach numbers to plot over. Defaults to np.linspace(0, 25, 500).
            alpha (array): Array of angles of attack to plot over (rad). Defaults to np.linspace(0, 4, 5)*np.pi/180.
        """

        CA = []
        CN = []
        COP = []

        for j in range(len(alpha)):
            #For each angle of attack
            CA.append([])
            CN.append([])
            COP.append([])

            for i in range(len(Mach)):
                #For each Mach number
                CA[j].append(self.CA(Mach[i], alpha[j]))
                CN[j].append(self.CN(Mach[i], alpha[j]))
                COP[j].append(self.COP(Mach[i], alpha[j]))

        #Create figure
        fig, axs = plt.subplots(2, 2)

        #Convert back to deg
        alpha = alpha*180/np.pi

        #Plot
        for i in range(len(alpha)):
            axs[0,0].plot(Mach, CA[i], label=f"alpha = {alpha[i]} deg")
            axs[0,1].plot(Mach, CN[i], label=f"alpha = {alpha[i]} deg")
            axs[1,0].plot(Mach, COP[i], label=f"alpha = {alpha[i]} deg")

        #Aesthetics
        axs[0,0].set_xlabel("Mach")
        axs[0,0].set_ylabel("CA")
        axs[0,0].grid()
        axs[0,0].legend()
        
        axs[0,1].set_xlabel("Mach")
        axs[0,1].set_ylabel("CN")
        axs[0,1].grid()
        axs[0,1].legend()

        axs[1,0].set_xlabel("Mach")
        axs[1,0].set_ylabel("COP (m)")
        axs[1,0].grid()
        axs[1,0].legend()

        plt.show()

    @staticmethod
    def from_lists(CA_list, CN_list, COP_list, Mach_list, alpha_list, ref_area, pitch_damping_coefficient = 0, roll_damping_coefficient = 0, error={"CA":1.0,"CN":1.0,"COP":1.0}):
        """Takes in 1D lists of data, and converts them into 2D arrays so they can be used for 2D interpolation.

        Args:
            CA_list (array, 1D): List of CA data at each Mach and alpha.
            CN_list (array, 1D): List of CN data at each Mach and alpha
            COP_list (array, 1D): List of COP data (m) at each mach and alph.
            Mach_list (array, 1D): List of Mach numbers for each data point.
            alpha_list (array, 1D): List of angles of attack (rad) for each data point.
            ref_area (array, 1D): Reference area used to normalise coefficients (m^2).
            pitch_damping_coefficient (int, optional): Pitch damping coefficient, defined by moment = C * ρ * ω^2. Defaults to 0.
            roll_damping_coefficient (int, optional): Roll damping coefficient, defined by moment = C * ρ * ω^2. Defaults to 0.
            error (dict, optional): Used for running stochastic analyses. Defaults to {"CA":1.0,"CN":1.0,"COP":1.0}.

        Returns:
            AeroData: AeroData object.
        """
        
        #Convert into the right shapes
        Mach_unique = np.unique(Mach_list)
        alpha_unique = np.unique(alpha_list)

        CA_grid = np.reshape(CA_list, (len(alpha_unique), len(Mach_unique)))
        CN_grid = np.reshape(CN_list, (len(alpha_unique), len(Mach_unique)))
        COP_grid = np.reshape(COP_list, (len(alpha_unique), len(Mach_unique)))

        return AeroData(CA_grid, CN_grid, COP_grid, Mach_unique, alpha_unique, ref_area, pitch_damping_coefficient, roll_damping_coefficient, error)

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

        #Convert COP from inches to m
        COP_raw = np.array(COP_raw) * 0.0254

        return AeroData.from_lists(CA_raw, CN_raw, COP_raw, Mach_raw, alpha_raw, ref_area, pitch_damping_coefficient, roll_damping_coefficient, error)

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