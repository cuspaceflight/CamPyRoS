'''
Suspected problems:
My post-shock flow temperatures seem very small
NASA FORTRAN code shows stuff on the order of 470K at station 1, down to around 256K at station 15.

I did some testing, and I think this might just be because the NASA rocket data is runing at extremely high Mach numbers (e.g. the first peak is at Mach 4)
The Martlet4 trajectory we have only hits like Mach 1 to 2, which might be too low for any significant aerodynamic heating.


'''


import numpy as np
import matplotlib.pyplot as plt
import thermo.mixture, json, scipy.integrate, scipy.optimize

from ambiance import Atmosphere
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, i2airspeed, pos_i2alt

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
    air = thermo.mixture.Mixture('air', T=T, P=P)    
    return air.Cp

def R_air():
    #Gas constant for air
    return 287

def gamma_air(T=298, P=1e-5):
    return 1.4

def Pr_air(T, P):
    air = thermo.mixture.Mixture('air', T=T, P=P)    
    return air.Pr

def k_air(T, P):
    air = thermo.mixture.Mixture('air', T=T, P=P)  
    return air.k

def mu_air(T, P):
    air = thermo.mixture.Mixture('air', T=T, P=P)  
    return air.mu

#Olibque shockwave functions, modified from: https://gist.github.com/gusgordon/3fa0a80e767a34ffb8b112c8630c5484
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

#Heat Transfer Analysis
class TangentOgive:
    def __init__(self, xprime, yprime):
        #https://arc.aiaa.org/doi/pdf/10.2514/3.62081 used for nomenclature
        self.xprime = xprime    #Longitudinal dimension
        self.yprime = yprime    #Base radius
        
        self.R = (xprime**2 + yprime**2)/(2*yprime)
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
        #i = 1 to 15
        assert i>=1 and i<=15, "i refers to stations 1-15, it cannot the less than 1 or more than 15"
        if i<=11:
            return 2 * self.R * np.sin((i-1)*self.dtheta/2) * np.sin(self.phi(i))
        else:
            return 2 * self.R * np.sin((10)*self.dtheta/2) * np.sin(self.phi(11))

    def S(self, i):
        #i = 1 to 15
        assert i>=1 and i<=15, "i refers to stations 1-15, it cannot the less than 1 or more than 15"
        return self.R * (i-1) * self.dtheta

class HeatTransfer:
    '''
    Notes
    ----------
    - Assumes that the angle of attack is always zero

    '''
    def __init__(self, tangent_ogive, trajectory_data, rocket):
        self.tangent_ogive = tangent_ogive
        self.trajectory_data = trajectory_data
        self.trajectory_dict = self.trajectory_data.to_dict(orient="list")
        self.rocket = rocket

        #Timestep index
        self.i = 0
        
        #Arrays to store the data at each discretised point on the nose cone (1 to 15), and at each timestep
        self.M = np.zeros([15, len(self.trajectory_dict["time"])])        #Local Mach number
        self.P = np.zeros([15, len(self.trajectory_dict["time"])])        #Local pressure
        starting_temperature = Atmosphere(pos_i2alt(self.trajectory_dict["pos_i"][0])).temperature[0] #Assume the nose cone starts with ambient temperature
        self.Tw = np.full([15, len(self.trajectory_dict["time"])], starting_temperature)              #Wall temperature - for now assume that wall temperature is constant
        self.Te = np.zeros([15, len(self.trajectory_dict["time"])])                                   #Temperature at the edge of the boundary layer
        self.Tstar = np.zeros([15, len(self.trajectory_dict["time"])])                                #T* as defined in the paper
        self.Hstar_function = np.zeros([15, len(self.trajectory_dict["time"])])                       #This array is used to minimise number of calculations for the integration needed in H*(x)

        #Arrays to store the useful data at the end
        self.q_lam = np.zeros([15, len(self.trajectory_dict["time"])])                #Laminar boundary layer
        self.q_turb = np.zeros([15, len(self.trajectory_dict["time"])])               #Turbunulent boundary layer
        self.q0_hemispherical_nose = np.zeros([15, len(self.trajectory_dict["time"])])#At the stagnation point for a rocket with a hemispherical nose cone - used as a reference point

    def step(self):
        '''I THINK THE (0) VALUES IN THE PAPER ARE FOR A POST NORMAL-SHOCK FLOW, NOT OBLIQUE-SHOCK'''
        '''I THINK THIS IS BECAUSE IT'S REFERING TO A SPHERICAL NOSE CONE AS A REFERENCE, WHICH WOULD HAVE A NORMAL SHOCK I THINK'''

        #Get altitude:
        alt = pos_i2alt(self.trajectory_dict["pos_i"][self.i])

        #Get ambient conditions:
        Pinf = Atmosphere(alt).pressure[0]
        Tinf = Atmosphere(alt).temperature[0]
        rhoinf = Atmosphere(alt).density[0]

        #Get the freestream velocity and Mach number
        Vinf = np.linalg.norm(i2airspeed(self.trajectory_dict["pos_i"][self.i], self.trajectory_dict["vel_i"][self.i], self.rocket.launch_site, self.trajectory_dict["time"][self.i]))
        Minf = Vinf/Atmosphere(alt).speed_of_sound[0]

        #Check if we're supersonic - if so we'll have a shock wave
        if Minf > 1:
            #For an oblique shock (tangent ogive nose cone)
            oblique_shock_data = cone_shock(self.tangent_ogive.theta, Minf, Tinf, Pinf, rhoinf) 
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
                    hw = cp_air() * self.Tw[j, self.i]
                    h0 = cp_air() * normal_T0S

                    #Prandtl numbers and specific heat capacities
                    Pre = Pr_air(self.Te[j, self.i], self.P[j, self.i])         

                    #'Reference' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3
                    hstar = (he + hw)/2 + 0.22*(Pre**0.5)*(h0 - hw)
                    self.Tstar[j, self.i] = hstar/cp_air()
                    Prstar = Pr_air(self.Tstar[j, self.i], self.P[j, self.i])

                    #'Recovery' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3 - I think these are the wall enthalpies for zero heat transfer
                    hrec_lam_boundary = he*(1-Prstar**(1/2)) + h0*(Prstar**(1/2))
                    hrec_turb_boundary = he*(1-Prstar**(1/3)) + h0*(Prstar**(1/3))

                    #Get H*(x) - WARNING: I'm a bit unsure about this section, in particular I'm not sure about the integral bit
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
                    Hstar = (rhostar * V * r)/(rhostar0 * Vinf) / (integral**0.5)

                    #Get H*(0)
                    RN = 0.3048      #Let RN = 1 ft = 0.3048m, as it recommends using that as a reference value (although apparently it shouldn't matter?)
                    dVdx0 = (2**0.5)/RN * ((normal_P0S - Pinf)/normal_rho0S)**0.5
                    Hstar0 = ( ((2*rhostar/rhostar0)*dVdx0 )/(Vinf * mustar/mustar0) )**0.5 * (2)**0.5

                    #Laminar heat transfer rate, normalised by that for a hemispherical nosecone
                    kstar = k_air(T = self.Tstar[j, self.i], P = self.P[j, self.i])
                    kstar0 = k_air(T = self.Tstar[j, self.i], P = normal_P0S)                  
                    Cpw = cp_air()
                    Cpw0 = cp_air()

                    #Equation (13) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081 - wasn't sure which 'hrec' to use here but I think it's the laminar one
                    qxq0_lam = (kstar * Hstar * (hrec_lam_boundary - hw) * Cpw0)/(kstar0 * Hstar0 * (h0 - hw) * Cpw)

                    #Stagnation point heat transfer rate for a hemispherical nosecone
                    Pr0 = Pr_air(normal_T0S, normal_P0S)  
                    rhow = rho02rho(oblique_rho0S, self.M[j, self.i]) 
                    muw = mu_air(self.Tw[j, self.i], self.P[j, self.i])
                    rho = rho02rho(oblique_rho0S, self.M[j, self.i])
                    mu = mu_air(self.Te[j, self.i], self.P[j, self.i])

                    #Equation (29) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    self.q0_hemispherical_nose[j, self.i] = 0.76*9.81*Pr0**(-0.6) * (rhow*muw)**0.1 * (rho*mu)**0.4 * (h0 - hw) * dVdx0**0.5

                    #Now we can find the absolute laminar heat transfer rates, in W/m^2
                    self.q_lam[j, self.i] = qxq0_lam * self.q0_hemispherical_nose[j, self.i]

                    #Turbulent heat transfer rate - using Equation (20) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    Cpstar0 = cp_air()
                    self.q_turb[j, self.i] = ( 0.03*9.81**(1/3) * (2**0.2) * kstar**(2/3) * (rhostar*V)**0.8 * (1 - Prstar**(1/3)*he + Prstar**(1/3)*h0 - hw) )/(mustar**(7/15) * Cpstar0**(2/3) * self.tangent_ogive.S(j+1)**0.2)               
                    
                    #print("i={} station={} he = {:.2f} kJ/kg/K hw = {:.2f} kJ/kg/K h0 = {:.2f} kJ/kg/K hinf = {} alt = {:.2f} m".format(self.i, j+1, he/1000, hw/1000, h0/1000, Tinf*cp_air()/1000, alt))
                    #print("i={} station={} Te={:.2f} Tw={:.2f} T0={:.2f} Tinf={} q_turb={} alt={:.2f} m".format(self.i, j+1, self.Te[j, self.i], self.Tw[j, self.i], normal_T0S, Tinf, self.q_turb[j, self.i], alt))
                    print("i={} station={} q_lam={:.2f} kW/m^2 q_turb={:.2f} kW/m^2 alt={:.2f} m t={:.2f} s".format(self.i, j+1, self.q_lam[j, self.i]/1000, self.q_turb[j, self.i]/1000, alt, self.trajectory_dict["time"][self.i]))


            else:
                print("Subsonic flow post-shock (Minf = {:.2f}, MS = {:.2f}), skipping step number {}".format(Minf, oblique_MS, self.i))

        else:
            print("Subsonic flow, skipping step number {}".format(self.i))

        
        self.i = self.i + 1

    def run(self, iterations = 300):
        for i in range(iterations):
            self.step()

    def to_json(self, directory="aero_heating_output.json"):
        dict = {"q_lam" : self.q_lam.tolist(), 
                "q_turb" : self.q_turb.tolist(), 
                "q0_hemispherical_nose" : self.q0_hemispherical_nose.tolist(), 
                "M" : self.M.tolist(), 
                "P" : self.P.tolist(),
                "Tw" : self.Tw.tolist(),
                "Te" : self.Te.tolist(),
                "Tstar" : self.Tstar.tolist(),
                "Hstar_function" : self.Hstar_function.tolist()}

        with open(directory, "w") as write_file:
            json.dump(dict, write_file)

    def from_json(self, directory):
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
        self.Hstar_function = np.array(dict["Hstar_function"])

    def plot_heat_transfer_rates(self, station_number=10, imax=None):
        assert station_number <= 15 and station_number >= 1, "Station number must be between 1 and 15 (inclusive)"
        q_lam = self.q_lam[station_number - 1, :]
        q_turb = self.q_turb[station_number - 1, :]
        
        trajectory_dict = self.trajectory_data.to_dict(orient="list")

        time = self.trajectory_dict["time"]
        alt = np.zeros(len(time))
        for i in range(len(time)):
            alt[i] = pos_i2alt(self.trajectory_dict["pos_i"][i])

        fig, axs = plt.subplots(2, 1)

        if imax == None:
            imax = len(time)

        axs[0].set_title("Heat transfer rates at station {}".format(int(station_number)))
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Heat transfer rate (W/m^2)")
        axs[0].plot(time[:imax], q_lam[:imax], label="Laminar heat transfer rate")
        axs[0].plot(time[:imax], q_turb[:imax], label="Turbulent heat transfer rate")
        axs[0].legend()
        axs[0].grid()

        axs[1].set_title("Altitude")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Altitude (m)")
        axs[1].plot(time[:imax], alt[:imax], color='orange')
        axs[1].grid()

        fig.tight_layout()
        plt.show()

        