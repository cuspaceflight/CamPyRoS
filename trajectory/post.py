import numpy as np
import thermo.mixture
import scipy.integrate, scipy.optimize

from ambiance import Atmosphere
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, pos_i2alt

#Properties of air
def cp_air(T=298, P=1e5):
    air = thermo.mixture.Mixture('air', T=T, P=P)    
    return air.Cp

def gamma_air(T=298, P=1e-5):
    return 1.4

def Pr_air(T, P):
    air = thermo.mixture.Mixture('air', T=T, P=P)    
    return air.Pr

def k_air(T, P):
    air = thermo.mixture.Mixture('air', T=T, P=P)  
    return air.k

#Shockwave functions, modified from: https://gist.github.com/gusgordon/3fa0a80e767a34ffb8b112c8630c5484
def taylor_maccoll(y, theta, gamma=1.4):
    # Taylor-Maccoll function
    # Source: https://www.grc.nasa.gov/www/k-12/airplane/coneflow.html
    v_r, v_theta = y
    dydt = [
        v_theta,
        (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2) 
    ]
    return dydt

def oblique_shock(theta, Ma, T, p, rho, gamma=1.4):
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

def cone_shock(cone_angle, Ma, T, p, rho):
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
        B, a, Ma2, T2, p2, rho2, v_x, v_y = oblique_shock(wedge_angle, Ma, T, p, rho)
        v_theta = v_y * np.cos(B) - v_x * np.sin(B)
        v_r = v_y * np.sin(B) + v_x * np.cos(B)
        y0 = [v_r, v_theta]
        thetas = np.linspace(B, cone_angle, 2000)

        sol = scipy.integrate.odeint(taylor_maccoll, y0, thetas)
        if sol[-1, 1] < 0:
            return B, a, Ma2, T2, p2, rho2, v_x, v_y

# Prandtl-Meyer Function
def prandtl_meyer(M, gamma=1.4):
    if M<1:
        raise ValueError("Cannot calculate the Prandtl-Meyer function of a flow with M<1")
    return float(np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1))) - np.arctan(np.sqrt(M ** 2 - 1)))

#Heat Transfer Analysis
class TangentOgive:
    def __init__(self, xprime, yprime):
        #https://arc.aiaa.org/doi/pdf/10.2514/3.62081 used for nomenclature
        self.xprime = xprime    #Longitudinal dimension
        self.yprime = yprime    #Base radius
        
        self.R = (xprime**2 + yprime**2)/(2*yprime)
        self.theta = np.arctan2(xprime, self.R - yprime)
        self.dtheta = 0.1*self.theta

    def phi(self, i):
        return self.theta - (i+1)*self.dtheta/2
    
    def r(self, i):
        return 2 * self.R * np.sin((i+1)*self.dtheta/2) * np.sin(self.phi(i))

    def S(self, i):
        return self.R * i * self.dtheta

class HeatTransfer:
    '''
    Notes
    ----------
    - Assumes that the angle of attack is always zero

    '''
    def __init__(self, tangent_ogive, simulation_output, rocket):
        self.tangent_ogive = tangent_ogive
        self.simulation_output = simulation_output.to_dict(orient="list")
        self.rocket = rocket

        #Timestep index
        self.i = 0
        
        #Arrays to store the local pressure and temperatures at each discertised point on the nose cone (15 points), and at each timestep
        self.P = np.zeros([15, len(self.simulation_output["time"])])
        self.Tw = self.P
        self.Te = self.P
        self.Tstar = self.P

    def step(self):
        #Altitude:
        self.alt = pos_i2alt(self.simulation_output["pos_i"][self.i])

        #Ambient properties
        self.Pinf = Atmosphere(self.alt).pressure[0]
        self.Tinf = Atmosphere(self.alt).temperature[0]
        self.rhoinf = Atmosphere(self.alt).density[0]

        #Check if we're supersonic - if so we'll have a shock wave
        self.V = np.linalg.norm(i2airspeed(self.simulation_output["pos_i"][self.i], self.simulation_output["vel_i"][self.i], self.rocket.launch_site, self.simulation_output["time"][self.i]))
        self.V = 600
        self.M = self.V/Atmosphere(self.alt).speed_of_sound[0]
        print("M = {}".format(self.M))

        if self.M > 1:
            #Find post-shock values
            print("theta = {}".format(self.tangent_ogive.theta*180/np.pi))
            shock_data = cone_shock(self.tangent_ogive.theta, self.M, self.Tinf, self.Pinf, self.rhoinf) 
            self.PS = shock_data[4]
            self.TS = shock_data[3]
            self.MS = shock_data[2]
            print("MS = {}".format(self.MS))

            #We can only do a Prandtl-Meyer expansion for supersonic flow:
            if self.MS > 1:
                #Prandtl-Meyer expansion from post-shockwave to each discretised point
                nu1 = prandtl_meyer(self.MS)
                theta1 = self.tangent_ogive.theta

                for j in range(11):
                    #Angle between the flow and the horizontal:
                    theta = self.tangent_ogive.theta - self.tangent_ogive.dtheta*j

                    #+mu characteristic: nu1 + theta1 = nu2 + theta2
                    nu = nu1 + theta1 - theta
                    print("nu{} = {}".format(j+1, nu))

                    #Check if we've exceeded nu-max, in which case we can't turn the flow any further
                    if nu > (np.pi/2)*(np.sqrt((gamma_air() + 1) / (gamma_air() - 1)) - 1):
                        print("Cannot turn flow any further at point {}".format(j+1))

                    #STUCK ON FINDING A WAY TO GET MACH NUMBER FROM NU
                    else:
                        def function_to_solve(M):
                            return prandtl_meyer(M) - nu

                        M = float(scipy.optimize.fmin_slsqp(function_to_solve, self.MS, bounds=[(1, np.inf)], iprint=0))
                        print("M{} = {}".format(j+1, M))
            
            else:
                print("Subsonic flow post-shock, skipping")






#Legacy
'''
        #Set up the initial conditions
        self.i = 0
        self.alt = pos_i2alt(self.simulation_output["pos_i"][self.i])   #Altitude

        self.P = Atmosphere(self.alt).pressure[0]       #Freestream pressure
        self.P0 = self.P                                #Stagnation pressure

        self.Te = Atmosphere(self.alt).temperature[0]   #Edge of boundary layer temperature
        self.Tw = self.Te                               #Wall temperature
        self.T0 = self.Te                               #Stagnation temperature

        self.he = cp_air()*self.Te                      #Edge of boundary layer enthalpy J/kg/K
        self.hw = cp_air()*self.Tw                      #Wall enthalpy J/kg/K
        self.h0 = self.he                               #Stagnation enthalpy

        self.Pre = Pr_air(self.Te, self.P)              #Edge of boundary layer Prandtl number

        #"Reference boundary layer values"
        self.hstar = (self.he + self.hw)/2 + 0.22*(self.Pre**0.5)*(self.h0 - self.hw)
        self.Tstar = self.hstar/cp_air()
        self.Prstar = Pr_air(self.Tstar, self.P)

        self.hrec_lam = self.he*(1-self.Prstar**(1/2)) + self.h0*(self.Prstar**(1/2))
        self.hrec_turb = self.he*(1-self.Prstar**(1/3)) + self.h0*(self.Prstar**(1/3))

        #Heat transfer rates
        qdot0 = k_air(self.Tstar, self.P)
'''