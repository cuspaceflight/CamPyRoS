'''
Implementation of NASA program NQLDW019. All values are stored in SI units unless otherwise stated. 

Reference material:
-------------------
[1] - HEATING ON SOUNDING ROCKET TANGENT OGIVE NOSES (NASA) - https://arc.aiaa.org/doi/pdf/10.2514/3.62081
[2] - TANGENT OGIVE NOSE AERODYNAMIC HEATING PROGRAM - NQLDW019 (NASA) - https://ntrs.nasa.gov/citations/19730063810

- There seems to be a typo in Equation (20) in Reference [1] (there's a missing bracket). The correct version seems to be present in Reference [2].

General Information:
-----------------------------
- Station 1 is the nosecone tip, 11 is at the nosecone base, and 12-15 are beyond the base.
- If the rocket is subsonic, or the post-oblique-shock flow is subsonic, the simulation skips the steps (NaN will usually be stored for the data for these steps, so it doesn't show up in graphs).

Assumptions:
-------------------
- If a variable wall temperature is used, the nosecone is modelled as a very simple lumped mass for temperature rises.
- Uniform wall temperature throughout the nose cone.
- Zero angle of attack. (if non-zero angle of attack is important, it can be implemented relatively simply and is explained in https://ntrs.nasa.gov/citations/19730063810)
- Constant Cp and Cv (hence constant gamma)

Known issues:
-------------
- The thermo module seems to have trouble finding the viscosity at some point - I think in post normal-shockwave conditions.
- I have not checked my model for wall temperature rise against any real data or the NASA examples.
- The current oblique shock functions were obtained from GitHub, so are technically copyrighted. I've added a custom ones which aren't currently used, but I haven't tested it as of yet.

Things I wasn't sure of:
-------------------------
- For the lumped mass wall temperature rise model, I improvised a way to deal with the infinite qdot at the nose tip (but am not sure how valid it is)
    I simply assumed the heat transfer rates at Station 1 (the nose tip) to be the same as those at Station 2

- Calculating for H*(0) - Equation (18) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
    I think that the rho(x) and mu(x) in Equation (18) are just rho(0) and mu(0), since they're evaluated at 'x=0'. This seemed to give the right values.
- Calculating H*(x) - Equation (17) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
    I think I did it right but the integration was a bit confusing
- Some equations only work in Imperial units. I believe I've taken care of them all, but if problems arise, that might be worth looking into

Nomenclature that is normally used:
-----------------------------------
T - Temperature (K)
rho - Density (kg/m^3)
P - pressure (Pa)
q or qdot - Heat transfer rate per unit area (W/m^2)
Q or Qdot - Heat transfer rate (W)

R - Gas constant (J/kg/K)
Cp or cp - Specific heat capacity at constant pressure (J/kg/K)
gamma - Ratio of specific heats (Cp/Cv)
Pr - Prandtl number
mu - Viscosity (Pa s)
k - Thermal conductivity (W/m/K, I think)

V - Velocity (m/s)
M - Mach number
Re - Reynolds number
nu - Prandtl-Meyer function evaluated at a given Mach and gamma

H - As defined in Equations (17) and (18) of Reference [1]
RN - Hemispherical nosecone radius (m)
dVdx0 - As defined in Equation (19) or Reference [1], units of (1/s) if I recall correctly

Subscripts:
-----------
e or x - "Local" value. This usually means it's taken at the edge of the boundary layer.
inf - Freestream (i.e. atmospheric) value.
ref or star - At the 'reference' enthalpy (usually marked with a star, e.g. T*, in the NASA documents). I'm not sure but I think this might be a sort of average boundary layer temperature.
0 - At the stagnation point for a hemispherical nose cone.
w - At wall temperature and local pressure.
rec - At 'recovery enthalpy', which is the same thing as at the 'adiabatic wall temperature' I believe.
turb - With a turbulent boundary layer
lam - With a laminar boundary layer.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import thermo.mixture, matplotlib.widgets, json, scipy.integrate, scipy.optimize
import gas_dynamics as gd

from ambiance import Atmosphere
from .transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, pos_i2alt



#Compressible flow functions
def prandtl_meyer(M, gamma=1.4):
    """Prandtl-Meyer function

    Parameters
    ----------
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        nu (Prandtl-Meyer function evaluated at the given Mach and gamma)

    """
    if M<1:
        raise ValueError("Cannot calculate the Prandtl-Meyer function of a flow with M < 1")

    return float(np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1))) - np.arctan(np.sqrt(M ** 2 - 1)))

def nu2mach(nu, gamma=1.4):
    """Inverse of the Prandtl-Meyer function
    Notes
    ----------
    Calculated using a polynomial approximation, described in http://www.pdas.com/pm.pdf

    Parameters
    ----------
    nu : float
        Value of the Prandtl-Meyer function
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Mach number corresponding to the given value of the Prandtl Meyer function

    """

    if gamma != 1.4:
        raise ValueError("This function will only work for gamma = 1.4")

    nuinf = (6**0.5 -1) * np.pi/2 
    y = (nu/nuinf)**(2/3)
    A = 1.3604
    B = 0.0962
    C = -0.5127
    D = -0.6722
    E = -0.3278

    return (1 + A*y + B*y**2 + C*y**3)/(1 + D*y + E*y**2)

def p2p0(P, M, gamma=1.4):
    """Returns static pressure from stagnation pressure, Mach number, and ratio of specific heats

    Parameters
    ----------
    P : float
        Static pressure
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Stagnation pressure

    """
    return P*(1 + (gamma - 1)/2 * M**2)**(gamma/(gamma - 1))

def p02p(P0, M, gamma=1.4):
    """Returns static pressure from stagnation pressure, Mach number, and ratio of specific heats

    Parameters
    ----------
    P0 : float
        Stagnation pressure
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Static pressure

    """
    return P0*(1 + (gamma - 1)/2 * M**2)**(-gamma/(gamma - 1))

def T2T0(T, M, gamma=1.4):
    """Returns stagnation temperature from static temperature, Mach number, and ratio of specific heats

    Parameters
    ----------
    T : float
        Static temperature
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Stagnation temperature

    """
    return T*(1 + (gamma - 1)/2 * M**2)

def T02T(T0, M, gamma=1.4):
    """Returns static temperature from stagnation temperature, Mach number, and ratio of specific heats

    Parameters
    ----------
    T0 : float
        Stagnation temperature
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Static temperature

    """
    return T0 * (1 + (gamma - 1)/2 * M**2)**(-1)

def rho2rho0(rho, M, gamma=1.4):
    """Returns stagnation density from static density, Mach number, and ratio of specific heats

    Parameters
    ----------
    T : float
        Static density
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Stagnation density

    """
    return rho * (1 + (gamma - 1)/2 * M**2)**(1/(gamma-1))

def rho02rho(rho0, M, gamma=1.4):
    """Returns static density from stagnation density, Mach number, and ratio of specific heats

    Parameters
    ----------
    rho0 : float
        Stagnation density
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Static density

    """
    return rho0 * (1 + (gamma - 1)/2 * M**2)**(-1/(gamma-1))

def pressure_ratio_to_mach(p_over_p0, gamma=1.4):
    """Get Mach number from the static to stagnation pressure ratio 

    Parameters
    ----------
    p_over_p0 : float
        Static pressure divided by stagnation pressure
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    float
        Mach number

    """
    return ( (2/(gamma-1)) * (p_over_p0**( (gamma-1)/-gamma) - 1) )**0.5

def normal_shock(M, gamma=1.4):
    """Normal shock wave calculator

    Parameters
    ----------
    M : float
        Mach number
    gamma : float, optional
        Ratio of specific heats (cp / cv). Defaults to 1.4.

    Returns
    -------
    numpy ndarray
        Returns array of floats in the following order: [MS, PS/P, TS/T, rhoS/rho]

    """
    MS = ((1 + 0.5*(gamma-1)*M**2 ) / (gamma*M**2 - 0.5*(gamma-1)))**0.5 
    PSoverP = 1 + 2*gamma/(gamma+1)*(M**2 - 1)
    TSoverT = (gamma-1)/(gamma+1)**2 * 2/M**2 * (1 + 0.5*(gamma-1)*M**2)*(2*gamma/(gamma-1) * M**2 - 1)
    rhoSoverrho = (gamma+1)*M**2 / (2*(1 + 0.5*(gamma-1)*M**2))

    return np.array([MS, PSoverP, TSoverT, rhoSoverrho])


#Properties of air
def cp_air(T=298, P=1e5):
    '''Specific heat capacity of air at constant pressure (J/kg/K)'''
    #air = thermo.mixture.Mixture('air', T=T, P=P)    
    #return air.Cp
    return 1005

def R_air():
    '''Gas constant for air (J/kg/K)'''
    return 287

def gamma_air(T=298, P=1e-5):
    '''Ratio of specific heats (Cp/Cv) for air'''
    return 1.4

def Pr_air(T, P):
    '''Prandtl number for air'''
    air = thermo.mixture.Mixture('air', T=T, P=P)    
    return air.Pr
    #return 0.71

def k_air(T, P):
    '''Thermal conductivity of air (W/m/K)'''
    air = thermo.mixture.Mixture('air', T=T, P=P)  
    return air.k
    #return 	26.24e-3

def mu_air(T, P):
    '''Viscosity of air (Pa s)'''
    air = thermo.mixture.Mixture('air', T=T, P=P)
    return air.mu
    #return 1.81e-5
    

#Olique shockwave functions, modified from: https://gist.github.com/gusgordon/3fa0a80e767a34ffb8b112c8630c5484
def taylor_maccoll(y, theta, gamma=1.4):
    # Taylor-Maccoll function
    # Source: https://www.grc.nasa.gov/www/k-12/airplane/coneflow.html
    v_r, v_theta = y
    dydt = [
        v_theta,
        (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2) 
    ]
    return dydt

def oblique_shock_github(theta, Ma, T, p, rho, gamma=1.4):
    """
    Computes the weak oblique shock resulting from supersonic
    flow impinging on a wedge in 2 dimensional flow.
    
    Inputs:
     - theta is the angle of the wedge in radians.
     - Ma, T, p, and rho are the Mach number, temperature (K),
       pressure (Pa), and density (kg/m^3) of the flow.
     - gamma is the ratio of specific heats. Defaults
       to air's typical value of 1.4.
    
    Returns:
     - shock angle in radians
     - resultant flow direction in radians
     - respectively, Mach number, temperature, pressure, density,
       and velocity components downstream of shock.
    
    Source: https://www.grc.nasa.gov/WWW/K-12/airplane/oblique.html
    """
    x = np.tan(theta)
    for B in np.arange(1, 500) * np.pi/1000:
        r = 2 / np.tan(B) * (Ma**2 * np.sin(B)**2 - 1) / (Ma**2 * (gamma + np.cos(2 * B)) + 2)
        if r > x:
            break
    cot_a = np.tan(B) * ((gamma + 1) * Ma ** 2 / (2 * (Ma ** 2 * np.sin(B) ** 2 - 1)) - 1)
    a = np.arctan(1 / cot_a)

    Ma2 = 1 / np.sin(B - theta) * np.sqrt((1 + (gamma - 1)/2 * Ma**2 * np.sin(B)**2) / (gamma * Ma**2 * np.sin(B)**2 - (gamma - 1)/2))

    h = Ma ** 2 * np.sin(B) ** 2
    T2 = T * (2 * gamma * h - (gamma - 1)) * ((gamma - 1) * h + 2) / ((gamma + 1) ** 2 * h)
    p2 = p * (2 * gamma * h - (gamma - 1)) / (gamma + 1)
    rho2 = rho * ((gamma + 1) * h) / ((gamma - 1) * h + 2)

    v2 = Ma2 * (gamma * 287 * T2)**0.5 
    v_x = v2 * np.cos(a)
    v_y = v2 * np.sin(a)
    return B, a, Ma2, T2, p2, rho2, v_x, v_y

def cone_shock_github(cone_angle, Ma, T, p, rho):
    """
    Computes properties of the conical oblique shock resulting
    from supersonic flow impinging on a cone in 3 dimensional flow.
    Inputs:
     - cone_angle is the half-angle of the 3D cone in radians.
     - Ma, T, p, and rho are the Mach number, temperature (K),
       pressure (Pa), and density (kg/m^3) of the flow.
    Returns:
     - shock angle in radians
     - flow redirection amount in radians
     - respectively, Mach number, temperature, pressure, density,
       and velocity components downstream of shock.
    Source: https://www.grc.nasa.gov/www/k-12/airplane/coneflow.html
    """

    wedge_angles = np.linspace(cone_angle, 0, 300)

    for wedge_angle in wedge_angles:
        B, a, Ma2, T2, p2, rho2, v_x, v_y = oblique_shock_github(wedge_angle, Ma, T, p, rho)
        v_theta = v_y * np.cos(B) - v_x * np.sin(B)
        v_r = v_y * np.sin(B) + v_x * np.cos(B)
        y0 = [v_r, v_theta]
        thetas = np.linspace(B, cone_angle, 2000)

        sol = scipy.integrate.odeint(taylor_maccoll, y0, thetas)
        if sol[-1, 1] < 0:
            return B, a, Ma2, T2, p2, rho2, v_x, v_y

#Custom oblique shock functions using gas_dynamics:
def oblique_shock(theta, Ma, T, p, rho, gamma=1.4, R=287):
    beta = gd.shocks.shocks.shock_angle(Ma, theta * 180/np.pi) * np.pi/180

    #From CUED 3A3 Databook
    PS = p*(1 + 2*gamma/(gamma+1) * (Ma**2 * np.sin(beta)**2 - 1))
    rhoS = rho*((gamma+1)*Ma**2*np.sin(beta)**2)/(2*(1 + (gamma-1)/2 * Ma**2 * np.sin(beta)**2))
    MS = ((1 + (gamma-1)/2 * Ma**2*np.sin(beta)**2)/(gamma*Ma**2 * np.sin(beta)**2 - (gamma-1)/2))**0.5 / np.sin(beta - theta)

    #Ideal gas - p = rho R T
    TS = PS/(rhoS*R)

    return 0, 0, MS, TS, PS, rhoS, 0, 0

#Aerodynamic heating analysis
class TangentOgive:
    '''
    Object used to store the geometry of a tangent ogive nose cone.

    Inputs
    -------
    xprime : float
        Longitudinal dimension (tip-to-base distance) (m)
    yprime : float
        Base radius (m)

    '''
    def __init__(self, xprime, yprime):
        #https://arc.aiaa.org/doi/pdf/10.2514/3.62081 used for nomenclature
        self.xprime = xprime    #Longitudinal dimension
        self.yprime = yprime    #Base radius
        
        self.R = (xprime**2 + yprime**2)/(2*yprime)         #Ogive radius
        self.theta = np.arctan2(xprime, self.R - yprime)
        self.dtheta = 0.1*self.theta

        #Each point (1 to 15) and its distance along the nose cone surface from the nose tip
        self.S_array = np.zeros(15)
        for i in range(len(self.S_array)):
            self.S_array[i] = self.S(i+1)

    def phi(self, i):
        #i = 1 to 15
        assert i>=1 and i<=15, "i refers to stations 1-15, it cannot the less than 1 or more than 15"
        return self.theta - (i-1)*self.dtheta/2
    
    def r(self, i):
        '''
        Local nosecone radius (y-dimension) at station

        Inputs
        -------
        i : int
            Station number (1-15)

        Returns
        -------
        float
            Local nosecone radius (m)
        '''
        #i = 1 to 15
        assert i>=1 and i<=15, "i refers to stations 1-15, it cannot the less than 1 or more than 15"
        if i<=11:
            return 2 * self.R * np.sin((i-1)*self.dtheta/2) * np.sin(self.phi(i))
        else:
            return 2 * self.R * np.sin((10)*self.dtheta/2) * np.sin(self.phi(11))

    def S(self, i):
        '''
        Distance from 

        Inputs
        -------
        i : int
            Station number (1-15)

        Returns
        --------
        float
            Distance along the nosecone surface (m), from the nosecone tip to station i.
        '''
        #i = 1 to 15
        assert i>=1 and i<=15, "i refers to stations 1-15, it cannot the less than 1 or more than 15"
        return self.R * (i-1) * self.dtheta

class AeroHeatingAnalysis:
    '''
    Object used to run aerodynamic heating analyses

    Inputs
    -------
    tangent_ogive : TangentOgive
        TangentOgive object specifying the nosecone geometry
    trajectory_data : dict or pandas DataFrame
        Data on the rocket's trajectory, needs to have "pos_i", "vel_i" and "time".
    rocket : Rocket
        Rocket object. It's only needed to get LaunchSite data for coordinate transforms.
    fixed_wall_temperature : bool
        If True, the wall temperature is fixed to its starting value. Otherwise a simple model is used to model its temperature change.
    starting_temperature : float, optional
        Temperature that the nose cone starts with (K). Defaults to None, in which case the rocket starts with the local atmospheric temperature. 
    nosecone_mass : float, optional
        Mass of the nosecone (kg) - used to find its heat capacity. Only needed if you're modelling variable temperatures
    specific_heat_capacity : float, optional
        Specific heat capacity of the nosecone (J/kg/K). Defaults to an approximate value for aluminium.
    turbulent_transition_Rex : float, optional
        Local Reynolds number at which the boundary layer transition from laminar to turbulent. Defaults to 7.5e6.

    Attributes
    ----------   
    i : int
        Current index in the timestep array.

    tangent_ogive : TangentOgive
        A TangentOgive object containing nosecone geometry data
    trajectory_data : dict or pandas DataFrame
        Contains trajectory data.
    trajectory_dict : dict
        Same data trajectory_data, but converted to a dictionary if it wasn't already one.
    rocket : Rocket
        Rocket object used to run the simulation. Its only purpose is to get LaunchSite data for coordinate transforms.
    fixed_wall_temperature : bool
        If True, the wall temperature is fixed to its starting value. Otherwise a simple model is used to model its temperature change.
    turbulent_transition_Rex : float, optional
        Local Reynolds number at which the boundary layer transition from laminar to turbulent. Defaults to 7.5e6.
    heat_capacity : float
        Heat capacity the nosecone (J/K). Is caclulated by doing specific_heat_capacity*nosecone_mass.
    
    M : numpy ndarray
        Local Mach number at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    P : numpy ndarray
        Local static pressure at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Te : numpy ndarray
        Temperature at the edge of the boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Tw : numpy ndarray
        Wall temperature at each timestep. Local static pressure at each station and timestep. Has dimensions (N) where N is the length of the "time" array in trajectory_dict.
    Trec_lam : numpy ndarray
        Adiabatic wall tempearture (also known as the 'recovery temperature') for a laminar boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Trec_turb : numpy ndarray
        Adiabatic wall tempearture (also known as the 'recovery temperature') for a turbulent boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Hstar_function : numpy ndarray
        Function that needs to be integrated to get H*(x), as defined in the NASA documents. This is not equal to H*(x) itself. This is stored because it's needed for integration. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Rex : numpy ndarray
        Local Reynolds number at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    
    q_lam : numpy ndarray
        Heat transfer rate (W/m^2) with a laminar boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    q_turb : numpy ndarray
        Heat transfer rate (W/m^2) with a turbulent boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    q0_hemispherical_nose : numpy ndarray
        Heat transfer rate (W/m^2) at the stagnation point at each timestep, if we were using a hemispherical nosecone. Has dimensions (N) where N is the length of the "time" array in trajectory_dict.
    
    '''
    def __init__(self, tangent_ogive, trajectory_data, rocket, 
                 fixed_wall_temperature = True,
                 starting_temperature = None, 
                 nosecone_mass = None, 
                 specific_heat_capacity = 900, 
                 turbulent_transition_Rex = 7.5e6):

        self.tangent_ogive = tangent_ogive
        self.trajectory_data = trajectory_data
        self.rocket = rocket
        self.turbulent_transition_Rex = turbulent_transition_Rex
        self.fixed_wall_temperature = fixed_wall_temperature

        #Convert the data into a dictionary if it isn't already one:
        if type(self.trajectory_data) is dict:
            self.trajectory_dict = self.trajectory_data
        else: 
            self.trajectory_dict = self.trajectory_data.to_dict(orient="list")

        #If we want a variable wall temperature:
        if self.fixed_wall_temperature == False:
            assert nosecone_mass != None, "You need to input a value for the nosecone mass if you want to model a variable wall temperature"
            self.heat_capacity = specific_heat_capacity * nosecone_mass

        #Timestep index:
        self.i = 0
        
        #Arrays to store the fluid properties at each discretised point on the nose cone (1 to 15), and at each timestep
        self.M = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))        #Local Mach number
        self.P = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))        #Local pressure

        #Initialise the wall temperature:
        if starting_temperature == None:
            starting_temperature = Atmosphere(pos_i2alt(self.trajectory_dict["pos_i"][0], self.trajectory_dict["time"][0])).temperature[0]           #Assume the nose cone starts with ambient temperature

        if self.fixed_wall_temperature == True:
            self.Tw = np.full(len(self.trajectory_dict["time"]), starting_temperature)
        else:
            self.Tw = np.full(len(self.trajectory_dict["time"]), float("NaN"))                                          
            self.Tw[0] = starting_temperature

        self.Te = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                                         #Temperature at the edge of the boundary layer
        self.Tstar = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                                      #T* as defined in the paper
        self.Trec_lam = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                                   #Temperature corresponding to hrec_lam
        self.Trec_turb = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                                  #Temperature corresponding to hrec_turb
        self.Hstar_function = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                             #This array is used to minimise number of calculations for the integration needed in H*(x)
        self.Rex = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                                        #Local Reynolds number

        #Arrays to store the heat transfer rates
        self.q_lam = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                     #Laminar boundary layer
        self.q_turb = np.full([15, len(self.trajectory_dict["time"])], float("NaN"))                    #Turbunulent boundary layer
        self.q0_hemispherical_nose = np.full(len(self.trajectory_dict["time"]), float("NaN"))     #At the stagnation point for a rocket with a hemispherical nose cone - used as a reference point

    def step(self, print_style=None):
        '''
        Performs one step of the aerodynamic analysis, starting from the current value of self.i.

        Inputs:
        -------
        print_style : str
            Options for print style:
            None - nothing is printed
            "FORTRAN" - same output as the examples in https://ntrs.nasa.gov/citations/19730063810, printing in the Imperial units listed
            "metric" - outputs useful data in metric units
        '''

        #Get altitude:
        alt = pos_i2alt(self.trajectory_dict["pos_i"][self.i], self.trajectory_dict["time"][self.i])

        #Get ambient conditions:
        Pinf = Atmosphere(alt).pressure[0]
        Tinf = Atmosphere(alt).temperature[0]
        rhoinf = Atmosphere(alt).density[0]

        #Get the freestream velocity and Mach number
        Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][self.i], self.trajectory_dict["vel_i"][self.i], self.rocket.launch_site, self.trajectory_dict["time"][self.i]))
        Minf = Vinf/Atmosphere(alt).speed_of_sound[0]

        if print_style=="FORTRAN":
            print("")
            print("FREE STREAM CONDITIONS")
            print("XMINF={:<10}     VINFY={:.4e}         GAMINF={:.4e}       RHOINF={:.4e}".format(0, 3.28084*Vinf, gamma_air(), 0.00194032*rhoinf))
            print("HINFY={:.4e}     PINF ={:.4e} (ATMOS) PINFY ={:.4e} (PSF)".format(0.000429923*cp_air()*Tinf, Pinf/101325, 0.0208854*Pinf))
            print("TINFY={:.4e}".format(Tinf))
            print("")

        if print_style=="metric":
            print("")
            print("SUBCRIPTS:")
            print("0 or STAG  : At the stagnation point for a hemispherical nose")
            print("REF        : At 'reference' enthalpy and local pressure - I think this is like an average-ish boundary layer enthalpy")
            print("REC        : At 'recovery' enthalpy and local pressure - I believe this is the wall enthalpy at which no heat transfer takes place")
            print("W          : At the wall temperature and local pressure")
            print("INF        : Freestream (i.e. atmospheric) property")
            print("LAM        : With a laminar boundary layer")
            print("TURB       : With a turbulent boundary layer")
            print("")
            print("FREE STREAM CONDITIONS")
            print("ALT ={:06.2f} km    TINF={:06.2f} K    PINF={:06.2f} kPa    RHOINF={:06.2f} kg/m^3".format(alt/1000, Tinf, Pinf/1000, rhoinf))
            print("VINF={:06.2f} m/s   MINF={:06.2f}".format(Vinf, Minf))
            print("")

        #Check if we're supersonic - if so we'll have a shock wave
        if Minf > 1:
            #For an oblique shock (tangent ogive nose cone)
            oblique_shock_data = cone_shock_github(self.tangent_ogive.theta, Minf, Tinf, Pinf, rhoinf) 
            oblique_PS = oblique_shock_data[4]
            oblique_TS = oblique_shock_data[3]
            oblique_MS = oblique_shock_data[2]
            oblique_rhoS = oblique_shock_data[5]

            oblique_P0S = p2p0(oblique_PS, oblique_MS)
            oblique_T0S = T2T0(oblique_TS, oblique_MS)
            oblique_rho0S = rho2rho0(oblique_rhoS, oblique_MS)

            #For a normal shock (hemispherical nosecone)
            normal_shock_data = normal_shock(Minf)
            normal_MS = normal_shock_data[0]
            normal_PS = normal_shock_data[1]*Pinf
            normal_TS = normal_shock_data[2]*Tinf
            normal_rhoS = normal_shock_data[3]*rhoinf

            normal_P0S = p2p0(normal_PS, normal_MS)
            normal_T0S = T2T0(normal_TS, normal_MS)
            normal_rho0S = rho2rho0(normal_rhoS, normal_MS)

            #Stagnation point heat transfer rate for a hemispherical nosecone
            Pr0 = Pr_air(normal_T0S, normal_P0S)  
            h0 = cp_air() * normal_T0S
            hw = cp_air() * self.Tw[self.i]
            mu0 = mu_air(normal_T0S, normal_P0S)
            rhow0 = normal_P0S/(R_air()*self.Tw[self.i])             #p = rho * R * T (ideal gas)
            muw0 = mu_air(self.Tw[self.i], normal_P0S)

            RN = 0.3048      #Let RN = 1 ft = 0.3048m, as it recommends using that as a reference value (although apparently it shouldn't matter?)
            dVdx0 = (2**0.5)/RN * ((normal_P0S - Pinf)/normal_rho0S)**0.5

            #Equation (29) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
            #Note that the equation only works in Imperial units, and requires you to specify density in slugs/ft^3, which is NOT lbm/ft^3
            #Metric density (kg/m^3) --> Imperial density (slugs/ft^3): Multiply by 0.00194032
            #Metric viscosity  (Pa s) --> Imperial viscosity (lbf sec/ft^2): Divide by 47.880259
            #Metric enthalpy (J/kg/s) --> Imperial enthalpy (Btu/lbm): Multiply by 0.000429923
            #Note that 'g', the acceleration of gravity, is equal to 32.1740 ft/s^2
            self.q0_hemispherical_nose[self.i] = 0.76*32.1740*Pr0**(-0.6) * (0.00194032*rhow0*muw0/47.880259)**0.1 * (0.00194032*normal_rho0S*mu0/47.880259)**0.4 * (0.000429923*h0 - 0.000429923*hw) * dVdx0**0.5

            #Now convert from Imperial heat transfer rate (Btu/ft^2/s) --> Metric heat transfer rate (W/m^2): Divide by 0.000088055
            self.q0_hemispherical_nose[self.i] = self.q0_hemispherical_nose[self.i]/0.000088055

            if print_style == "FORTRAN":
                print("")
                print("STAGNATION POINT DATA FOR SPHERICAL NOSE")
                print("HREF0 ={:<10}     TREF0 ={:<10}   VISCR0={:<10}   TKREF0={:<10}".format(0, 0, 0, 0))
                print("ZREF0 ={:<10}     PRREF0={:<10}   CPREF0={:<10}   RHOR0 ={:<10}".format(0, 0, 0, 0))
                print("CPCVR0={:.4e}     RN    ={:.4e}   T0    ={:.4e}".format(gamma_air(), RN/0.3048, normal_T0S))
                print("P0    ={:.4e}     RHO0  ={:.4e}   SR0   ={:<10}   TK0   ={:<10}".format(normal_P0S/101325, 0.00194032*normal_rho0S, 0, 0))
                print("VISC0 ={:.4e}     DVDX0 ={:.4e}   Z0    ={:<10}   CP0   ={:.4e}".format(mu0/47.880259, dVdx0, 0, 0.000429923*cp_air()))
                print("A0    ={:<10}     TW0   ={:.4e}   VISCW0={:.4e}   HW0   ={:.4e}".format(0, self.Tw[self.i], muw0/47.880259, 0.000429923*hw))
                print("")
                print("CPW0  ={:.4e}     PR0   ={:.4e}".format(0.000429923*cp_air(), Pr0))
                print("QSTPT ={:.4e}     = NOSE STAGNATION POINT HEAT RATE".format(0.000088055*self.q0_hemispherical_nose[self.i]))
                print("H0    ={:.4e}     HT    ={:<10}   RHOW0={:.4e}".format(0.000429923*h0, 0, 0.00194032*rhow0))
                print("")

            if print_style == "metric":
                print("")
                print("STAGNATION POINT DATA FOR SPHERICAL NOSE")
                print("P0   ={:06.2f} kPa    T0   ={:06.2f} K       RHO0={:06.2f} kg/m^3".format(normal_P0S/1000, normal_T0S, normal_rho0S))
                print("TW   ={:06.2f} K      RHOW0={:06.2f} kg/m^3".format(self.Tw[self.i], rhow0))
                print("QSTAG={:06.2f} kW/m^2".format(self.q0_hemispherical_nose[self.i]/1000))
                print("")


            #Prandtl-Meyer expansion (only possible for supersonic flow):
            if oblique_MS > 1:
                #Get values at the nose cone tip:
                nu1 = prandtl_meyer(oblique_MS)
                theta1 = self.tangent_ogive.theta

                #Prandtl-Meyer expansion from post-shockwave to each discretised point
                for j in range(10):
                    #Angle between the flow and the horizontal:
                    theta = self.tangent_ogive.theta - self.tangent_ogive.dtheta*j

                    #Across a +mu characteristic: nu1 + theta1 = nu2 + theta2
                    nu = nu1 + theta1 - theta

                    #Check if we've exceeded nu_max, in which case we can't turn the flow any further
                    if nu > (np.pi/2)*(np.sqrt((gamma_air() + 1) / (gamma_air() - 1)) - 1):
                        raise ValueError("Cannot turn flow any further at nosecone position {}, exceeded nu_max. Flow will have seperated (which is not yet implemented). Stopping simulation.".format(j+1))
                    
                    #Record the local Mach number and pressure
                    self.M[j, self.i] = nu2mach(nu)
                    self.P[j, self.i] = p02p(oblique_P0S, self.M[j, self.i])
                
                #Expand for the last few points using Equations (1) - (6) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                for j in [10, 11, 12, 13, 14]:
                    if j>=10 and j<=13:
                        self.P[j, self.i] = (Pinf + self.P[j - 1, self.i])/2
                    elif j==14:
                        self.P[j, self.i] = Pinf
                    self.M[j, self.i] = pressure_ratio_to_mach(self.P[j, self.i]/oblique_P0S)


                #Now deal with the heat transfer itself
                for j in range(15):
                    #Edge of boundary layer temperature - i.e. flow temperature post-shock and after Prandtl-Meyer expansion
                    self.Te[j, self.i] = T02T(oblique_T0S, self.M[j, self.i]) 

                    #Enthalpies
                    he = cp_air() * self.Te[j, self.i]

                    #Prandtl numbers and specific heat capacities
                    Pre = Pr_air(self.Te[j, self.i], self.P[j, self.i])         

                    #'Reference' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3
                    hstar = (he + hw)/2 + 0.22*(Pre**0.5)*(h0 - hw)
                    self.Tstar[j, self.i] = hstar/cp_air()
                    Prstar = Pr_air(self.Tstar[j, self.i], self.P[j, self.i])

                    #'Recovery' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3 - I think these are the wall enthalpies for zero heat transfer
                    hrec_lam = he*(1-Prstar**(1/2)) + h0*(Prstar**(1/2))
                    hrec_turb = he*(1-Prstar**(1/3)) + h0*(Prstar**(1/3))
                    self.Trec_lam[j, self.i] = hrec_lam/cp_air()
                    self.Trec_turb[j, self.i] = hrec_turb/cp_air()

                    #Get H*(x) - I'm not sure about if I did the integral bit right
                    rhostar0 = normal_P0S / (R_air() * self.Tstar[j, self.i])
                    mustar0 = mu_air(T=self.Tstar[j, self.i], P = normal_P0S)

                    rhostar = self.P[j, self.i] / (R_air() * self.Tstar[j, self.i])    
                    mustar = mu_air(T=self.Tstar[j, self.i], P = self.P[j, self.i])
                    
                    r = self.tangent_ogive.r(j+1)
                    V = (gamma_air() * R_air() * T02T(oblique_T0S, self.M[j, self.i]))**0.5 * self.M[j, self.i]  

                    self.Hstar_function[j, self.i] = (rhostar*mustar*V* r**2) / (rhostar0 * mustar0 * Vinf)    

                    #Get the integral bit of H*(x) using trapezium rule
                    integral = np.trapz(self.Hstar_function[0:j+1, self.i], self.tangent_ogive.S_array[0:j+1])

                    #Equation (17) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    if j == 0:
                        Hstar = np.inf
                    else:
                        Hstar = (rhostar * V * r)/(rhostar0 * Vinf) / (integral**0.5)

                    #Get H*(0) - Equation (18) - it seems like the (x) values in Equation (18) are actually (0) values
                    #It seems weird that they still included them though, since they end up cancelling out
                    #Hstar0 = ( ((2*rhostar/rhostar0)*dVdx0 )/(Vinf * mustar/mustar0) )**0.5 * (2)**0.5 
                    Hstar0 = (2*dVdx0/Vinf)**0.5 * (2**0.5)

                    #Laminar heat transfer rate, normalised by that for a hemispherical nosecone
                    kstar = k_air(T = self.Tstar[j, self.i], P = self.P[j, self.i])
                    kstar0 = k_air(T = self.Tstar[j, self.i], P = normal_P0S)                  
                    Cpw = cp_air()
                    Cpw0 = cp_air()

                    #Equation (13) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081 - wasn't sure which 'hrec' to use here but I think it's the laminar one
                    qxq0_lam = (kstar * Hstar * (hrec_lam - hw) * Cpw0)/(kstar0 * Hstar0 * (h0 - hw) * Cpw)

                    #Now we can find the absolute laminar heat transfer rates, in W/m^2
                    self.q_lam[j, self.i] = qxq0_lam * self.q0_hemispherical_nose[self.i]

                    #Turbulent heat transfer rate - using Equation (20) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    #THERE LOOKS LIKE THERE'S A TYPO IN THE EQUATION! It should be {1 - (Pr*)^(1/3)}*he, not 1 - {(Pr*)^(1/3)}*he
                    #For the correct version see page Eq (20) on page 14 of https://ntrs.nasa.gov/citations/19730063810

                    #Note that the equation only works in Imperial units, and requires you to specify density in slugs/ft^3, which is NOT lbm/ft^3
                    #Density (kg/m^3) --> Density (slugs/ft^3): Multiply by 0.00194032
                    #Viscosity  (Pa s) --> Viscosity (lbf sec/ft^2): Divide by 47.880259
                    #Enthalpy (J/kg/s) --> Enthalpy (Btu/lbm): Multiply by 0.000429923
                    #Thermal conductivity (W/m/K) --> Thermal conductivity (Btu/ft/s/K): Multiply by 0.000288894658
                    #Velocity (m/s) ---> Velocity (ft/s): Multiply by 3.28084
                    #Note that 'g', the acceleration of gravity, is equal to 32.1740 ft/s^2

                    Cpstar0 = cp_air()
                    if j == 0:
                        self.q_turb[j, self.i] = np.inf
                    else:
                        self.q_turb[j, self.i] = ( 0.03*32.1740**(1/3) * (2**0.2) * (0.000288894658*kstar)**(2/3) * (0.00194032*rhostar*3.28084*V)**0.8 * ((1 - Prstar**(1/3))*0.000429923*he + Prstar**(1/3)*0.000429923*h0 - 0.000429923*hw) )/((mustar/47.880259)**(7/15) * (0.000429923*Cpstar0)**(2/3) * (3.28084*self.tangent_ogive.S(j+1))**0.2)               
                    
                    #Now convert from Imperial heat transfer rate (Btu/ft^2/s) --> Metric heat transfer rate (W/m^2): Divide by 0.000088055
                    self.q_turb[j, self.i] = self.q_turb[j, self.i]/0.000088055

                    #Local Reynolds number, Re(x), using Equation (25) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    rho = self.P[j, self.i]/(R_air() * self.Te[j, self.i])  #Ideal gas law: p = rho*R*T, rho = p/(RT)
                    mu = mu_air(self.Te[j, self.i], self.P[j, self.i])
                    self.Rex[j, self.i] = rho * V * self.tangent_ogive.S_array[j]/mu

                    #FORTRAN style output:
                    if print_style=="FORTRAN":
                        print("")
                        print("WALL, REFERENCE AND EXTERNAL-TO-BOUNDARY-LAYER FLOW PROPERTIES AT STATION = {}".format(j+1))
                        print("HW    ={:.4e}    CPW   ={:.4e}   HREFX={:<10}    PRREFX={:<10}".format(0.000429923*hw, 0.000429923*Cpw, 0, 0))
                        print("TKREFX={:<10}    VISCRX={:<10}   RHORX={:<10}    TREFX ={:<10}".format(0, 0, 0, 0))
                        print("ZREFX ={:<10}    CPCVRX={:<10}   PX   ={:.4e}    TX    ={:.4e}".format(0, 0, self.P[j, self.i]/101325, self.Te[j, self.i]))
                        print("TKX   ={:<10}    VISCX ={:.4e}   PRX  ={:.4e}    ZX    ={:<10}".format(0, mu_air(self.Te[j, self.i], self.P[j, self.i])/47.880259, Pre, 0))
                        print("SRX   ={:<10}    HX    ={:.4e}   VX   ={:.4e}    CPCVX ={:.4e}".format(0, 0.000429923*he, 3.28084*V, gamma_air()))
                        print("AAX   ={:<10}    RHOX  ={:.4e}   XM   ={:<10}    CPX   ={:.4e}".format(0, 0.00194032*rho02rho(oblique_rho0S, self.M[j, self.i]), 0, 0.000429923*cp_air()))
                        print("")
                        print("X = {:.3f}".format(3.28084*self.tangent_ogive.S_array[j]))
                        print("QLAM={:.3f}     QTURB={:.3f}     QLAM/QSTAG={:.3f}     QTURB/QSTAG={:.3f}".format(0.000088055*self.q_lam[j, self.i], 0.000088055*self.q_turb[j, self.i], self.q_lam[j, self.i]/self.q0_hemispherical_nose[self.i], self.q_turb[j, self.i]/self.q0_hemispherical_nose[self.i]))
                        print("")

                    if print_style=="metric":
                        print("")
                        print("WALL, REFERENCE AND EXTERNAL-TO-BOUNDARY-LAYER FLOW PROPERTIES AT STATION = {}".format(j+1))
                        print("X   ={:.6} m".format(self.tangent_ogive.S_array[j]))
                        print("PX  ={:.6} kPa        TX   ={:06.2f} K        RHOX      ={:06.2f} kg/m^3".format(self.P[j, self.i]/1000, self.Te[j, self.i], rho02rho(oblique_rho0S, self.M[j, self.i])))
                        print("TW  ={:.6} K          TREF ={:06.2f} K        TREC_LAM  ={:06.2f} K     TREC_TURB  ={:06.2f} K".format(self.Tw[self.i], self.Tstar[j, self.i], hrec_lam/cp_air(), hrec_lam/cp_air()))
                        print("QLAM={:.6} kW/m^2     QTURB={:06.2f} kW/m^2   QLAM/QSTAG={:06.2f}       QTURB/QSTAG={:06.2f}".format(self.q_lam[j, self.i]/1000, self.q_turb[j, self.i]/1000, self.q_lam[j, self.i]/self.q0_hemispherical_nose[self.i], self.q_turb[j, self.i]/self.q0_hemispherical_nose[self.i]))
                        print("")


                #Simple lumped mass model for increase in wall temperature:
                if self.fixed_wall_temperature == False:

                    #Points 12 - 15 are below the bottom of the nosecone, so we'll ignore them.
                    qdot_array = np.zeros(11)
                    qdotr_array = np.zeros(11) #Local q * local nosecone radius

                    for j in range(len(qdot_array)):
                        #The nose tip (Station 1) has q = infinity, so we'll ignore it for now
                        if j == 0:
                            pass

                        #Check if we have a laminar or turbulent boundary layer at each point:
                        else:
                            if self.Rex[j, self.i] < self.turbulent_transition_Rex:
                                #Laminar boundary layer
                                qdot_array[j] = self.q_lam[j, self.i]
                            else:
                                #Turbulent boundary layer
                                qdot_array[j] = self.q_turb[j, self.i]
                        
                        #qdot(x) * r(x)
                        qdotr_array[j] = qdot_array[j] * self.tangent_ogive.r(j+1)
                    
                    #Set the heat transfer rates at Station 1 (the nose tip) to be the same as that at Station 2
                    qdot_array[0] = qdot_array[1]
                    qdotr_array[0] = qdotr_array[1]

                    #Integrate to get the total heat transferred
                    Qdot_tot = 2*np.pi * np.trapz(qdotr_array, self.tangent_ogive.S_array[:11])    #Qdot = ∫qdot dA = ∫qdot (2πrdx) = 2π∫qdot r dx
                    Q_tot = Qdot_tot * (self.trajectory_dict["time"][self.i+1] - self.trajectory_dict["time"][self.i])  #Q = ∫Qdot dt, using left Riemann sum

                    #Get the change in temperature, and add it to the current temperature
                    dT = Q_tot/self.heat_capacity
                    self.Tw[self.i + 1] = self.Tw[self.i] + dT

                else:
                    #If using a fixed wall temperature:
                    self.Tw[self.i + 1] = self.Tw[self.i]

            else:
                if print_style != None:
                    print("Subsonic flow post-shock (Minf = {:.2f}, MS = {:.2f}), skipping step number {}".format(Minf, oblique_MS, self.i))

                #Wall temperature doesn't change:
                self.Tw[self.i + 1] = self.Tw[self.i]

        else:
            if print_style != None:
                print("Subsonic freestream flow, skipping step number {}".format(self.i))

            #Wall temperature doesn't change:
            self.Tw[self.i + 1] = self.Tw[self.i]
        
        self.i = self.i + 1

    def run(self, number_of_steps = None, starting_index = 0, print_style="minimal"):
        '''
        Runs the simulation for a set number of steps, starting from starting_index. Updates all of its attributes as it does so.
        
        Inputs:
        -------
        number_of_steps : int
            Number of steps you would like to perform. Defaults to None, in which case the programme goes through all available data in trajectory_data.
        starting_index : int
            The index in the "time" array that you want to start from. Note that you should always start from 0 if you're using a variable wall temperature (previous wall temperatures will affect the heat transfer rate, and hence affect future wall temperatures).
        print_style : str
            Options for print_style:
            None - Nothing is printed
            "minimal" - Minimalistic printing, printing progress every 10%, and the max. and min. wall temperature if a variable wall temperature is used.
        '''
        if number_of_steps == None:
            number_of_steps = len(self.trajectory_dict["time"]) - 1 - starting_index

        if self.fixed_wall_temperature == False and starting_index != 0:
            print("WARNING: You should normally start the simulation from starting_index = 0 if you're using a variable wall temperature. Doing otherwise may give inaccurate results or errors.")

        self.i = starting_index
        counter = 0

        while self.i-starting_index <= number_of_steps:
            if print_style=="minimal":
                if (self.i - starting_index) % (int(number_of_steps/10)) == 0:
                    print("{:.1f}% complete, i = {}".format(counter/10 * 100, self.i))
                    counter = counter + 1

            self.step()
        
        if self.fixed_wall_temperature == False and print_style=="minimal":
            print("Maximum wall temparature = {:.4f} °C".format(np.nanmax(self.Tw)-273.15))
            print("Minimum wall temperature = {:.4f} °C".format(np.nanmin(self.Tw)-273.15))

    def to_json(self, directory="aero_heating_output.json"):
        '''
        Outputs the current data stored in attributes to a .JSON file.

        Inputs
        ------
        directory : str
            Directory you want to save to .JSON file to.
        '''
        dict = {"q_lam" : self.q_lam.tolist(), 
                "q_turb" : self.q_turb.tolist(), 
                "q0_hemispherical_nose" : self.q0_hemispherical_nose.tolist(), 
                "M" : self.M.tolist(), 
                "P" : self.P.tolist(),
                "Tw" : self.Tw.tolist(),
                "Te" : self.Te.tolist(),
                "Tstar" : self.Tstar.tolist(),
                "Trec_lam" : self.Trec_lam.tolist(),
                "Trec_turb" : self.Trec_turb.tolist(),
                "Hstar_function" : self.Hstar_function.tolist(),
                "Rex" : self.Rex.tolist()}

        with open(directory, "w") as write_file:
            json.dump(dict, write_file)

        print("Exported data to {}".format(directory))

    def from_json(self, directory):
        '''
        Imports data from a .JSON file (which normally would have been produced using AeroHeatingAnalysis.to_json) and uses it to fill the data attributes.

        Inputs
        ------
        directory : str
            Directory to the .JSON file you want to import aerodynamic heating data from.
        '''
        with open(directory, "r") as read_file:
            dict = json.load(read_file)
        
        self.q_lam = np.array(dict["q_lam"])
        self.q_turb = np.array(dict["q_turb"])
        self.q0_hemispherical_nose = np.array(dict["q0_hemispherical_nose"])
        self.M = np.array(dict["M"])
        self.P = np.array(dict["P"])
        self.Tw = np.array(dict["Tw"])
        self.Te = np.array(dict["Te"])
        self.Tstar = np.array(dict["Tstar"])
        self.Trec_lam = np.array(dict["Trec_lam"])
        self.Trec_turb = np.array(dict["Trec_turb"])
        self.Hstar_function = np.array(dict["Hstar_function"])
        self.Rex = np.array(dict["Rex"])

    def plot_station(self, station_number=10, imax=None):
        '''
        Plots data at a given station on the nosecone.

        Inputs
        ------
        station_number : int
            Station number (1-15) to plot data at.
        imax : int
            Maximum index (of the trajectory_dict["time"] array) to plot to.
        '''
        assert station_number <= 15 and station_number >= 1, "Station number must be between 1 and 15 (inclusive)"

        time = self.trajectory_dict["time"]
        alt = np.zeros(len(time))
        for i in range(len(time)):
            alt[i] = pos_i2alt(self.trajectory_dict["pos_i"][i], self.trajectory_dict["time"][i])

        fig, axs = plt.subplots(2, 2)

        if imax == None:
            imax = len(time)

        fig.suptitle("Properties at Station {}".format(int(station_number)))
        axs[0,1].set_title("Heat transfer rates")
        axs[0,1].set_xlabel("Time (s)")
        axs[0,1].set_ylabel("Heat transfer rate (kW/m^2)")
        axs[0,1].plot(time[:imax], self.q_turb[station_number - 1, :imax]/1000, label=r"$\dot{q}_{turb}$")
        axs[0,1].plot(time[:imax], self.q_lam[station_number - 1, :imax]/1000, label=r"$\dot{q}_{lam}$")
        #axs[0,1].plot(time[:imax], self.q0_hemispherical_nose[:imax]/1000, label=r"$\dot{q}_{0}$")
        axs[0,1].legend()
        axs[0,1].grid()

        axs[1,1].set_title("Temperatures")
        axs[1,1].set_xlabel("Time (s)")
        axs[1,1].set_ylabel("Temperature (K)")
        axs[1,1].plot(time[:imax], self.Trec_turb[station_number - 1, :imax], label=r"$T_{rec, turb}$")
        axs[1,1].plot(time[:imax], self.Trec_lam[station_number - 1, :imax], label=r"$T_{rec, lam}$")
        axs[1,1].plot(time[:imax], self.Te[station_number - 1, :imax], label=r"$T_{e}$")
        axs[1,1].plot(time[:imax], self.Tw[:imax], label=r"$T_{w}$")
        axs[1,1].grid()
        axs[1,1].legend()

        axs[1,0].set_title("Local Reynolds Number")
        axs[1,0].set_xlabel("Time (s)")
        axs[1,0].set_ylabel("Re(x)")
        axs[1,0].plot(time[:imax], self.Rex[station_number - 1, :imax], color='green', label=r'$Re_{x}$')
        axs[1,0].plot(time[:imax], np.full(len(time[:imax]), self.turbulent_transition_Rex), color='green', linestyle="--", label=r'$Re_{transition}$')
        axs[1,0].set_yscale("log")
        axs[1,0].legend()
        axs[1,0].grid()

        axs[0,0].set_title("Altitude")
        axs[0,0].set_xlabel("Time (s)")
        axs[0,0].set_ylabel("Altitude (m)")
        axs[0,0].plot(time[:imax], alt[:imax], color='orange')
        axs[0,0].grid()

        fig.tight_layout()
        plt.show()

    def plot_heat_transfer(self, i=0, y_limit_scaling=1.5, automatic_rescaling=True):
        '''
        Plots heat transfer along the nosecone. Has an interactive slider that you can use to browse through different indexes of the trajectory_dict["time"] array.

        Inputs
        -------
        i : int
            The trajectory_dict["time"] index to start the plot at. Defaults to 0.
        y_limit_scaling : float
            Used to scale the y-limits on the plots. This is only used if automatic_rescaling = False.
        automatic_rescaling : bool
            If True, the axes scales are redrawn every time you move the slider to a new position. If False, the software attempts to find a set of axes limits that are acceptable at all datapoints.
        '''
        fig, axs = plt.subplots(2, 2)

        alt = pos_i2alt(self.trajectory_dict["pos_i"][i], self.trajectory_dict["time"][i])
        Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][i], self.trajectory_dict["vel_i"][i], self.rocket.launch_site, self.trajectory_dict["time"][i]))
        Minf = Vinf/Atmosphere(alt).speed_of_sound[0]
        fig.suptitle('i={} time = {:.2f} s alt={:.2f} km Minf={:.2f}'.format(i, self.trajectory_dict["time"][i], alt/1000, Minf))

        ogive_x = np.linspace(0, self.tangent_ogive.xprime, 15)
        ogive_y = (self.tangent_ogive.R**2 - (self.tangent_ogive.xprime - ogive_x)**2)**0.5 + self.tangent_ogive.yprime - self.tangent_ogive.R

        axs[0,0].set_title("Heat transfer rates")
        axs[0,0].set_xlabel("Station")
        axs[0,0].set_ylabel("Heat transfer rate (kW/m^2)")
        qlam_plot, = axs[0,0].plot(np.array(range(15))+1, self.q_lam[:, i]/1000, label = r"$\dot{q}_{lam}$")
        qturb_plot, = axs[0,0].plot(np.array(range(15))+1, self.q_turb[:, i]/1000, label = r"$\dot{q}_{turb}$")
        #q0_plot, = axs[0,0].plot(np.array(range(15))+1, np.full(15, self.q0_hemispherical_nose[i]/1000), label = r"$\dot{q}_{0}$")
        axs[0,0].legend(bbox_to_anchor=(-0.4, 1))
        axs[0,0].grid()

        axs[0,1].set_title("Temperatures")
        axs[0,1].set_xlabel("Station")
        axs[0,1].set_ylabel("Temperature (K)")
        Te_plot, = axs[0,1].plot(np.array(range(15))+1, self.Te[:, i], label=r"$T_e$")
        Tstar_plot, = axs[0,1].plot(np.array(range(15))+1, self.Tstar[:, i], label=r"$T*$")
        Tw_plot, = axs[0,1].plot(np.array(range(15))+1, np.full(15, self.Tw[i]), label=r"$T_w$")
        Tinf_plot, = axs[0,1].plot(np.array(range(15))+1, np.full(15, Atmosphere(alt).temperature[0]), label=r"$T_{inf}$")
        Trec_lam_plot, = axs[0,1].plot(np.array(range(15))+1, self.Trec_lam[:, i], label=r"$T_{rec, lam}$")
        Trec_turb_plot, = axs[0,1].plot(np.array(range(15))+1, self.Trec_turb[:, i], label=r"$T_{rec, turb}$")
        axs[0,1].legend(bbox_to_anchor=(1.05, 1))
        axs[0,1].grid() 

        axs[1,0].set_title("Local Reynolds numbers")
        axs[1,0].set_xlabel("Station")
        axs[1,0].set_ylabel("Reynolds number")
        axs[1,0].set_yscale("log")
        Re_plot, = axs[1,0].plot(np.array(range(15))+1, self.Rex[:, i], label=r"$Re_x$", color="green")
        Re_transition_plot, = axs[1,0].plot(np.array(range(15))+1, np.full(15, self.turbulent_transition_Rex), label=r"$Re_{transition}$", color="green", linestyle='--')
        axs[1,0].legend(bbox_to_anchor=(-0.4, 1))
        axs[1,0].grid()

        #Fixed axes whilst the slider is changed:
        if automatic_rescaling==False:
            axs[0,0].set_ylim([np.nanmin(self.q_lam[self.q_lam != 0])/y_limit_scaling/1000, y_limit_scaling*self.q0_hemispherical_nose[self.q0_hemispherical_nose != 0].max()/1000])
            axs[0,1].set_ylim([self.Te[self.Te>0].min()/y_limit_scaling, y_limit_scaling*self.Tstar.max()])
            axs[1,0].set_ylim([self.P[self.P>0].min()/y_limit_scaling/1000, y_limit_scaling*self.P.max()/1000])
        
        axs[1,1].set_title("Nosecone shape")
        axs[1,1].set_xlabel("x (m)")
        axs[1,1].set_ylabel("y (m)")
        axs[1,1].plot(ogive_x, ogive_y)
        axs[1,1].grid()
        axs[1,1].set_aspect('equal')
        axs[1,1].set_xlim(-0.3*ogive_x[-1], 1.1*ogive_x[-1])
        axs[1,1].set_ylim(-2*ogive_y[-1], 5*ogive_y[-1])

        fig.tight_layout()

        #Add a slider - modified from https://stackoverflow.com/questions/46325447/animated-interactive-plot-using-matplotlib
        slider_axis = plt.axes([0.25, 0, 0.50, 0.02])
        initial_value = i   #Initial slider value                          
        #Make a slider that goes from 0 to the maximum index available for our data
        slider = matplotlib.widgets.Slider(slider_axis, 'Index', 0, len(self.trajectory_dict["time"])-1, valinit=initial_value)

        def update(val):
            #Get the current value of the slider
            slider_value = slider.val
            index = int(slider_value)

            #Get current altitude
            alt = pos_i2alt(self.trajectory_dict["pos_i"][index], self.trajectory_dict["time"][index])

            #Update curves
            Rex_plot.set_ydata(self.Rex[:, index])
            Te_plot.set_ydata(self.Te[:, index])
            Tstar_plot.set_ydata(self.Tstar[:, index])
            Tw_plot.set_ydata(np.full(15, self.Tw[index]))
            Tinf_plot.set_ydata(np.full(15, Atmosphere(alt).temperature[0]))
            Trec_lam_plot.set_ydata(self.Trec_lam[:, index])
            Trec_turb_plot.set_ydata(self.Trec_turb[:, index])
            qlam_plot.set_ydata(self.q_lam[:, index]/1000)
            qturb_plot.set_ydata(self.q_turb[:, index]/1000)
            #q0_plot.set_ydata(np.full(15, self.q0_hemispherical_nose[index]/1000))
            
            #Rescale the limits if asked to
            if automatic_rescaling==True:
                for i in range(len(axs)):
                    for j in range(len(axs[i])):
                        axs[i,j].relim()
                        axs[i,j].autoscale_view()

            #Recreate the title
            Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][index], self.trajectory_dict["vel_i"][index], self.rocket.launch_site, self.trajectory_dict["time"][index]))
            Minf = Vinf/Atmosphere(alt).speed_of_sound[0]
            fig.suptitle('i={} time = {:.2f} s alt={:.2f} km Minf={:.2f}'.format(index, self.trajectory_dict["time"][index], alt/1000, Minf))

            #Redraw canvas while idle
            fig.canvas.draw_idle()


        #Call update function on slider value change
        slider.on_changed(update)

        plt.show()

    def plot_fluid_properties(self, i=0, y_limit_scaling=1.5, automatic_rescaling=True):
        '''
        Plots fluid properties at along the nosecone (Static pressures, temperatures, Mach numbers). Has an interactive slider that you can use to browse through different indexes of the trajectory_dict["time"] array.

        Inputs
        -------
        i : int
            The trajectory_dict["time"] index to start the plot at. Defaults to 0.
        y_limit_scaling : float
            Used to scale the y-limits on the plots. This is only used if automatic_rescaling = False.
        automatic_rescaling : bool
            If True, the axes scales are redrawn every time you move the slider to a new position. If False, the software attempts to find a set of axes limits that are acceptable at all datapoints.
        '''
        fig, axs = plt.subplots(2, 2)

        alt = pos_i2alt(self.trajectory_dict["pos_i"][i], self.trajectory_dict["time"][i])
        Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][i], self.trajectory_dict["vel_i"][i], self.rocket.launch_site, self.trajectory_dict["time"][i]))
        Minf = Vinf/Atmosphere(alt).speed_of_sound[0]
        fig.suptitle('i={} time = {:.2f} s alt={:.2f} km Minf={:.2f}'.format(i, self.trajectory_dict["time"][i], alt/1000, Minf))

        ogive_x = np.linspace(0, self.tangent_ogive.xprime, 15)
        ogive_y = (self.tangent_ogive.R**2 - (self.tangent_ogive.xprime - ogive_x)**2)**0.5 + self.tangent_ogive.yprime - self.tangent_ogive.R
        
        axs[0,0].set_title("Pressures")
        axs[0,0].set_xlabel("Station")
        axs[0,0].set_ylabel("Pressure (kPa)")
        pe_plot, = axs[0,0].plot(np.array(range(15))+1, self.P[:, i]/1000, label=r"$P_e$")
        pinf_plot, = axs[0,0].plot(np.array(range(15))+1, np.full(15, Atmosphere(alt).pressure[0]/1000), label=r"$P_{inf}$")
        axs[0,0].legend(bbox_to_anchor=(-0.4, 1))
        axs[0,0].grid()

        axs[0,1].set_title("Temperatures")
        axs[0,1].set_xlabel("Station")
        axs[0,1].set_ylabel("Temperature (K)")
        Te_plot, = axs[0,1].plot(np.array(range(15))+1, self.Te[:, i], label=r"$T_e$")
        Tstar_plot, = axs[0,1].plot(np.array(range(15))+1, self.Tstar[:, i], label=r"$T*$")
        Tw_plot, = axs[0,1].plot(np.array(range(15))+1, np.full(15, self.Tw[i]), label=r"$T_w$")
        Tinf_plot, = axs[0,1].plot(np.array(range(15))+1, np.full(15, Atmosphere(alt).temperature[0]), label=r"$T_{inf}$")
        Trec_lam_plot, = axs[0,1].plot(np.array(range(15))+1, self.Trec_lam[:, i], label=r"$T_{rec, lam}$")
        Trec_turb_plot, = axs[0,1].plot(np.array(range(15))+1, self.Trec_turb[:, i], label=r"$T_{rec, turb}$")
        axs[0,1].legend(bbox_to_anchor=(1.05, 1))
        axs[0,1].grid()

        axs[1,0].set_title("Mach numbers")
        axs[1,0].set_xlabel("Station")
        axs[1,0].set_ylabel("Mach number")
        M_plot, = axs[1,0].plot(np.array(range(15))+1, self.M[:, i], label=r"$M_e$")
        Minf_plot, = axs[1,0].plot(np.array(range(15))+1, np.full(15, Minf), label=r"$M_{inf}$")
        axs[1,0].legend(bbox_to_anchor=(-0.4, 1))
        axs[1,0].grid()

        if automatic_rescaling==False:
            axs[1,0].set_ylim([self.M[self.M>0].min()/y_limit_scaling, y_limit_scaling*self.M.max()])
            axs[0,1].set_ylim([self.Te[self.Te>0].min()/y_limit_scaling, y_limit_scaling*self.Tstar.max()])
            axs[0,0].set_ylim([self.P[self.P>0].min()/y_limit_scaling/1000, y_limit_scaling*self.P.max()/1000])

        axs[1,1].set_title("Nosecone shape")
        axs[1,1].set_xlabel("x (m)")
        axs[1,1].set_ylabel("y (m)")
        axs[1,1].plot(ogive_x, ogive_y)
        axs[1,1].grid()
        axs[1,1].set_aspect('equal')
        axs[1,1].set_xlim(-0.3*ogive_x[-1], 1.1*ogive_x[-1])
        axs[1,1].set_ylim(-2*ogive_y[-1], 5*ogive_y[-1])

        fig.tight_layout()

        #Add a slider - modified from https://stackoverflow.com/questions/46325447/animated-interactive-plot-using-matplotlib
        slider_axis = plt.axes([0.25, 0, 0.50, 0.02])
        initial_value = i   #Initial slider value                          
        #Make a slider that goes from 0 to the maximum index available for our data
        slider = matplotlib.widgets.Slider(slider_axis, 'Index', 0, len(self.trajectory_dict["time"])-1, valinit=initial_value)

        def update(val):
            #Get the current value of the slider
            slider_value = slider.val
            index = int(slider_value)

            #Get current altitude
            alt = pos_i2alt(self.trajectory_dict["pos_i"][index], self.trajectory_dict["time"][index])

            #Recreate the title
            Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][index], self.trajectory_dict["vel_i"][index], self.rocket.launch_site, self.trajectory_dict["time"][index]))
            Minf = Vinf/Atmosphere(alt).speed_of_sound[0]
            fig.suptitle('i={} time = {:.2f} s alt={:.2f} km Minf={:.2f}'.format(index, self.trajectory_dict["time"][index], alt/1000, Minf))

            #Update curves
            pe_plot.set_ydata(self.P[:, index]/1000)
            pinf_plot.set_ydata(np.full(15, Atmosphere(alt).pressure[0]/1000))
            Te_plot.set_ydata(self.Te[:, index])
            Tstar_plot.set_ydata(self.Tstar[:, index])
            Tw_plot.set_ydata(np.full(15, self.Tw[index]))
            Tinf_plot.set_ydata(np.full(15, Atmosphere(alt).temperature[0]))
            Trec_lam_plot.set_ydata(self.Trec_lam[:, index])
            Trec_turb_plot.set_ydata(self.Trec_turb[:, index])
            M_plot.set_ydata(self.M[:, index])
            Minf_plot.set_ydata(np.full(15, Minf))

            #Rescale the limits if asked to
            if automatic_rescaling==True:
                for i in range(len(axs)):
                    for j in range(len(axs[i])):
                        axs[i,j].relim()
                        axs[i,j].autoscale_view()

            #Redraw canvas while idle
            fig.canvas.draw_idle()

        #Call update function on slider value change
        slider.on_changed(update)

        plt.show()
