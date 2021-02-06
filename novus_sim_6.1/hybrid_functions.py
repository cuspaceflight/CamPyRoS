"""Contains additional functions used by motor_sim.py"""

### Joe Hunt updated 20/06/19 ###
### All units SI unless otherwise stated ###

__copyright__ = """

    Copyright 2019 Joe Hunt

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Modified by Daniel Gibbons 2021 (changes documented in commit history)

"""

import numpy as np
import scipy.optimize


def dyer_injector(cpres, inj_dia, lden, inj_pdrop, hl, manifold_P, vpres):
    """Models the mass flow rate of (initially liquid) n2o through a single
    injector orifice using the 2-phase model proposed by Dyer et al"""
    inj_pdrop_og = 0

    if inj_pdrop < 3*10**5:
        print('accuracy warning: injector pdrop so low that 2-phase Dyer model no longer applies.',
              'approximating with linear pdrop/mdot characteristic')
        inj_pdrop_og = inj_pdrop
        inj_pdrop = 3*10**5
    # get downstream spec. enthalpy and density
    h2, rho2 = chamber_vap(cpres)
    # single-phase incompressible mass flow rate:
    Cd = 0.6 #Waxman et al, adapted for square edged orifices
    mdot_spi = Cd*(((inj_dia/2)**2)*np.pi)*((2*lden*inj_pdrop)**(1/2))
    # mass flow rate by homogenous equilibrium model:
    mdot_hem = Cd*(((inj_dia/2)**2)*np.pi)*rho2*((2*-(hl-h2))**(1/2))
    if vpres < cpres:
        raise RuntimeError("injector pdrop lower than vapour pressure",
                           "2-phase Dyer model no longer applies!")
    # non-equilibrium parameter k (∝ ratio of bubble growth time and liquid residence time)
    k = ((manifold_P-cpres)/(vpres-cpres))**(1/2)
    # mass flow rate by Dyer model
    mdot_ox = ((k/(1+k))*mdot_spi)  +  ((1/(1+k))*mdot_hem)
    if inj_pdrop_og < 3*10**5 and inj_pdrop_og > 0:
        mdot_ox *= (inj_pdrop_og/(3*10**5))
    return mdot_ox



def c_star_lookup(cpres, OF, propep_data):
    """Looks up ratio of characteristic velocity from chamber pressure and OF
    ratio using propep data"""

    if cpres > 90E5 or cpres < 0:
        raise RuntimeError('chamber pressure out of propep data range!')
    if OF > 39 or OF < (1/39):
        raise RuntimeError('OF out of propep data range!')

    rounded_cpres = int(5*round((cpres/(10**5))/5))
    cpres_line = int(765*(((100-rounded_cpres)/5)-1)+9)
    rounded_oxpct = (round(2*((OF/(1+OF))*100)))/2
    oxpct_line = int((4*((97.5-rounded_oxpct)/0.5))+3)
    # note that propep data is in feet/s
    Cstar_fps = float((propep_data[cpres_line+oxpct_line-1].split()[4]))
    Cstar = Cstar_fps*0.3048 #convert to m/s
    return Cstar



def gamma_lookup(cpres, OF, propep_data):
    """Looks up ratio of specific heats from chamber pressure and OF ratio
    using propep data"""
    if cpres > 90E5 or cpres < 0:
        raise RuntimeError('chamber pressure out of propep data range!')
    if OF > 39 or OF < (1/39):
        raise RuntimeError('OF out of propep data range!')

    rounded_cpres = int(5*round((cpres/(10**5))/5))
    cpres_line = int(765*(((100-rounded_cpres)/5)-1)+9)
    rounded_oxpct = (round(2*((OF/(1+OF))*100)))/2
    oxpct_line = int((4*((97.5-rounded_oxpct)/0.5))+3)
    gamma = float((propep_data[cpres_line+oxpct_line-1].split()[1]))
    return gamma



def vapour_injector(inj_dia, vden, inj_pdrop):
    """Models the mass flow rate of single phase vapour-only through a single
    injector orifice"""
    cd = 0.65 # somewhat a guess
    mdot_spi = cd*(((inj_dia/2)**2)*np.pi)*((2*vden*inj_pdrop)**(1/2))
    return mdot_spi



def Z2_solve(temp1, Z1, vmass1, vmass2, gamma_n2o, zdat, pdat):
    """Finds current compressibility factor given the initial compressibility
    factor and the initial and current vapour masses"""

    def temp2_delta(Z2, Z1, vmass1, vmass2, gamma_n2o):
        """Function returns difference between vapour temperature as calculated
        by isentropic relation and that from compressibility factor data"""
        # new vapour temperature (isentropic relation and ideal gas law)
        temp2_isen = temp1*(((Z2*vmass2)/(Z1*vmass1))**(gamma_n2o-1))
        temp2_Z = temp_solve_Z(Z2, zdat, pdat) #temperature corresponding to specified Z2
        delta_Z2 = temp2_isen-temp2_Z
        return delta_Z2

    # find Z2 at which the two calculated temperatures are equal
    if (np.sign(temp2_delta(min(zdat), Z1, vmass1, vmass2, gamma_n2o))
            == np.sign(temp2_delta(max(zdat), Z1, vmass1, vmass2, gamma_n2o))):
        return 'numerical instability'
    else:
        Z2_solved = scipy.optimize.bisect(temp2_delta, min(zdat), max(zdat),
                                          args=tuple([Z1, vmass1, vmass2, gamma_n2o]))
        return Z2_solved



def ball_valve_K(Re, d1, d2, L):
    """Returns full-bore ball valve flow coefficient as a thick orifice"""
    if Re < 2500:
        K = (2.72+((d2/d1)**2)*((120/Re)-1))*(1-((d2/d1)**2))*(((d1/d2)**4)-1)
    else:
        K = (2.72+((d2/d1)**2)*(4000/Re))*(1-((d2/d1)**2))*(((d1/d2)**4)-1)
    K = K*(0.584+(0.0936/((L/d2)**(1.5)+0.225)))
    return K



def Nikuradse(Re):
    """Returns the friction factor, f, for a given Reynolds number, fitting the
    Nikuradse model"""
    f = ((0.0076*((3170/Re)**0.165))/(1+((3170/Re)**7)))+(16/Re)
    return f

def thermophys(temp):
    """Returns N2O liquid density, vapour density, latent heat of vaporization,
    dynamic viscosity and vapour pressure for input temperature.
    Uses polynomials from ESDU sheet 91022. All units SI."""

    if temp > 309.57 or temp < 183.15:
        raise ValueError('nitrous oxide temperature out of data range')

    lden = (452*(np.e**(1.72328*((1-(temp/309.57))**(1/3))+-0.8395*((1-(temp/309.57))**(2/3))
                        +0.5106*(1-(temp/309.57))+-0.10412*((1-(temp/309.57))**(4/3)))))

    vden = (452*(np.e**(-1.009*(((1/(temp/309.57))-1)**(1/3))
                        +-6.28792*(((1/(temp/309.57))-1)**(2/3))
                        +7.50332*((1/(temp/309.57))-1)
                        +-7.90463*(((1/(temp/309.57))-1)**(4/3))
                        +0.629427*(((1/(temp/309.57))-1)**(5/3)))))

    hl = ((-200+116.043*((1-(temp/309.57))**(1/3))+-917.225*((1-(temp/309.57))**(2/3))
           +794.779*(1-(temp/309.57))+-589.587*((1-(temp/309.57))**(4/3)))*1000)

    hg = ((-200+440.055*((1-(temp/309.57))**(1/3))+-459.701*((1-(temp/309.57))**(2/3))
           +434.081*(1-(temp/309.57))+-485.338*((1-(temp/309.57))**(4/3)))*1000)

    c = ((2.49973*(1+0.023454*((1-(temp/309.57))**(-1))+-3.80136*(1-(temp/309.57))
                   +13.0945*((1-(temp/309.57))**2)+-14.5180*((1-(temp/309.57))**3)))*1000)

    vpres = (7251000*(np.e**((1/(temp/309.57))*(-6.71893*(1-(temp/309.57))
             +1.35966*((1-(temp/309.57))**(3/2))+-1.3779*((1-(temp/309.57))**(5/2))
             +-4.051*(1-(temp/309.57))**5))))

    ldynvis = (0.0293423*np.e**((1.609*(((309.57-5.24)/(temp-5.24)-1)**(1/3)))
               +(2.0439*(((309.57-5.24)/(temp-5.24)-1)**(4/3)))))

    return (lden, vden, hl, hg, c, vpres, ldynvis)


def temp_solve_P(P):
    """Returns the temperature given the vapour pressure. Second argument is an
    initial guess temperature for Newton raphson solver"""
    # define difference between vapour pressure as a function of temperature
    # and the given pressure(ESDU 91022)
    def vpres_delta(temp, P):
        vpres_delta = ((7251000*(np.e**((1/(temp/309.57))*(-6.71893*(1-(temp/309.57))
                       +1.35966*((1-(temp/309.57))**(3/2))+-1.3779*((1-(temp/309.57))**(5/2))
                       +-4.051*(1-(temp/309.57))**5))))-P)
        return vpres_delta
    # solve for given pressure
    temp_solve = scipy.optimize.bisect(vpres_delta, 183.15, 309.57, args=tuple([P]))
    return temp_solve


def chamber_vap(P):
    """Returns N2O specific enthalpy of vapour and vapour density at chambe
    pressure, again, 2nd argument is guess of vapour temperature"""
    temp = temp_solve_P(P)
    hg, vden = thermophys(temp)[3], thermophys(temp)[1]
    return (hg, vden)


def compressibility_read(compressibility_data):
    """Return lists of pressure and compressibility factors from csv file"""
    pdat = []
    zdat = []
    next(compressibility_data)
    next(compressibility_data)
    for row in compressibility_data:
        pdat.append(float(row[0]))
        zdat.append(float(row[1]))
    return pdat, zdat

def temp_solve_Z(Z, zdat, pdat):
    """Returns the temperature given the compressibility factor"""
    P = np.interp(Z, zdat[::-1], pdat[::-1])
    temp_solve_Z_return = temp_solve_P(P)
    return temp_solve_Z_return




def mach_exit(gamma, NOZZLE_AREA_RATIO):
    """Returns the exit mach number by numerical solution"""

    def mach_error(m):
        """Finds the discrepancy between the non-dimensional mass flow
        rate (stagnation) found using Mach number relations and that
        calculated using current guess of exit conditions"""
        m_new = ((1/NOZZLE_AREA_RATIO)
                 *((((2/(gamma+1))*(1+(gamma-1)*(m**2)/2))**((gamma+1)/(gamma-1)))**(1/2)))
        return abs(m-m_new)

    mach_no_exit = float((scipy.optimize.minimize(mach_error, 4, tol=1e-9,
                                                  method='Powell')).x)
    return mach_no_exit



