"""Mass Models

Stores mass models

Notes
--------

- All "positions" of masses are measured relative to the tip of the rocket's nose
- All "times" are measured from the moment of ignition
"""

import scipy
import numpy as np
    
class CylindricalMassModel:
    """Simple cylindrical model of the rockets mass and moments of inertia.

    Note
    ----
    Assumes the rocket is a solid cylinder, constant volume, which has a mass that reduces with time (i.e. the density of the cylinder reducs)

    Parameters
    ----------
    mass : list
        Masses of the rocket at time after ignition /kg
    time : list
        Corresponding time for the mass /s
    l : float
        Length of rocket (cylinder) /m
    r : float
        Radius of rocket (cylinder) /m

    Attributes
    ----------
    mass : Scipy Interpolation Function
        Masses of the rocket at time after ignition, when called interpolates to desired time /kg
    time : list
        Corresponding time for the mass /s
    l : float
        Length of rocket (cylinder) /m
    r : float
        Radius of rocket (cylinder) /m

    """
    def __init__(self, mass, time, l, r):    
        self.mass_interp = scipy.interpolate.interp1d(time, mass)
        self.time = time
        self.l = l
        self.r = r

    def mass(self, time):  
        """Returns the mass at some time after igition

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        time : float
            Time since ignition /s

        Returns
        -------
        float
            Mass interpolated at time /lg

        """     
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.mass()")
        elif time < self.time[0]:
            return self.mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.mass_interp(time)
        else:
            return self.mass_interp(self.time[-1])

    def ixx(self, time):
        """Returns the xx moment of inertia at some time after igition

        Parameters
        ----------
        time : float
            Time since ignition /s

        Returns
        -------
        float
            xx moment of inertia /kgm^2

        """      
        if time<0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
        elif time < self.time[0]:
            return (1/2)* self.r**2 * self.mass(self.time[0])
        elif time < self.time[-1]:
            return (1/2)* self.r**2 * self.mass(time)
        else:
            return (1/2)* self.r**2 * self.mass(self.time[-1])

      
    def iyy(self, time):
        """Returns the yy moment of inertia at some time after igition

        Parameters
        ----------
        time : float
            Time since ignition /s

        Returns
        -------
        float
            yy moment of inertia /kgm^2

        """    
        if time < 0:
            raise ValueError("Tried to input negative time when using CylindricalMassModel.iyy() or .izz()")
        elif time < self.time[0]:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[0])
        elif time < self.time[-1]:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(time)
        else:
            return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(self.time[-1])
      
    def izz(self, time):
        """Returns the zz moment of inertia at some time after igition

        Parameters
        ----------
        time : float
            Time since ignition /s

        Returns
        -------
        float
            zz moment of inertia /kgm^2

        """    
        return self.iyy(time)
    
    def cog(self, time):
        """Returns the centre of gravity at some time after igition

        Parameters
        ----------
        time : float
            Time since ignition /s

        Returns
        -------
        float
            Centre of gravity /m

        """         
        return self.l/2

class LiquidFuel:
    """Mass model for the liquid in a cylindrical fuel tank

    Notes
    ----------
    - Ignore the mass of the vapour, this must be included seperately


    Parameters
    ----------


    Attributes
    ----------


    """
    def __init__(self, liq_den, liq_mass, tank_radius, pos_tank_bottom, time):  
        #Functions of time:
        self.liq_den = liq_den     
        self.liq_mass = liq_mass     
        self.time = time
        
        #Set up interpolated functions
        self.mass_interp = scipy.interpolate.interp1d(time, liq_mass)
        self.den_interp = scipy.interpolate.interp1d(time, liq_den)

        #Constants:
        self.tank_radius = tank_radius      #Radius of fuel tank
        self.pos_tank_bottom = pos_tank_bottom     #Distance between the rocket's nose tip and the bottom of the fuel tank

    def mass(self, time):
        #Mass of the liquid  
        if time<0:
            raise ValueError("Tried to input negative time when using LiquidTank.mass()")
        elif time < self.time[0]:
            return self.mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.mass_interp(time)
        else:
            return self.mass_interp(self.time[-1])

    def den(self, time):
        #Density of the liquid
        if time<0:
            raise ValueError("Tried to input negative time when using LiquidTank.den()")
        elif time < self.time[0]:
            return self.den_interp(self.time[0])
        elif time < self.time[-1]:
            return self.den_interp(time)
        else:
            return self.den_interp(self.time[-1])
            
    def vol(self, time):
        #Volume of liquid
        return self.mass(time)/self.den(time)

    def liq_height(self, time):
        #Height of the liquid relative to the bottom of the tank
        return self.vol(time)/(np.pi * self.tank_radius**2)

    def ixx(self, time):
        #Liquid is assumed to have no moment of inertia about the long axis (basically assumes it's inviscid so doesnt rotate with the rocket)
        return 0.0

    def iyy(self, time):
        #Assume a uniform solid cylinder
        return self.mass(time) * (self.tank_radius**2 / 4 + self.liq_height(time)**2 / 12)

    def izz(self, time):
        return self.iyy(time)

    def cog(self, time):
        #Distance of the liquid centre of gravity from the rocket nose tip
        return self.pos_tank_bottom - self.liq_height(time)/2

class SolidFuel:
    """Mass model for a solid fuel grain

    Notes
    ----------
    Assumes that the grain is an annular cylinder, constant outer radius, and an inner radius that increases as fuel is burnt up

    Parameters
    ----------


    Attributes
    ----------


    """
    def __init__(self, fuel_mass, fuel_density, r_out, length, pos_bottom, time):  
        #Functions of time
        self.fuel_mass = fuel_mass
        self.time = time

        #Constants
        self.fuel_density = fuel_density
        self.r_out = r_out
        self.length = length
        self.pos_bottom = pos_bottom    #Distance of the bottom of the fuel grain from the rocket nose tip

        #Set up interpolated functions
        self.mass_interp = scipy.interpolate.interp1d(time, fuel_mass)

        #Distance between the rocket nose tip and the centre of gravity of the fuel
        self._cog = self.pos_bottom + self.length/2

    def mass(self, time):
        #Mass of the solid fuel  
        if time<0:
            raise ValueError("Tried to input negative time when using SolidFuel.mass()")
        elif time < self.time[0]:
            return self.mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.mass_interp(time)
        else:
            return self.mass_interp(self.time[-1])
    
    def r_in(self, time):
        #Inner radius of the fuel grain
        return ( self.r_out**2 - self.mass(time)/(np.pi * self.fuel_density * self.length) )**0.5

    def ixx(self, time):
        a = (self.r_out + self.r_in(time))/2    #Average radius
        t = self.r_out - self.r_in(time)        #Thickness

        return self.mass(time) * (a**2 + (t**2)/4)

    def iyy(self, time):
        a = (self.r_out + self.r_in(time))/2    #Average radius
        t = self.r_out - self.r_in(time)        #Thickness

        return self.mass(time) * ((a**2)/2 + (t**2)/8 + (self.length**2)/12)

    def izz(self, time):
        return self.iyy(time)

    def cog(self, time=0):
        return self._cog

class CustomMass:
    """For storing custom mass data

    Notes
    ----------
    

    Parameters
    ----------


    Attributes
    ----------


    """

    def __init__(self, mass, ixx, iyy, izz, cog):
        self._mass = mass
        self._ixx = ixx
        self._iyy = iyy
        self._izz = izz

    def mass(self, time=0):
        return self._mass
    
    def ixx(self, time=0):
        return self._ixx

    def iyy(self, time=0):
        return self._iyy

    def izz(self, time=0):
        return self._izz

class HollowCylinder:
    """To help calculate moments of inertia for a hollow (i.e. annular) cylinder

    Notes
    ----------
    

    Parameters
    ----------


    Attributes
    ----------


    """

    def __init__(self, r_out, r_in, l, mass):
        self.r_out = r_out
        self.r_in = r_in
        self.l = l
        self.mass = mass

        self.t = r_out - r_in
        self.a = (r_out + r_in)/2

        self.ixx = self.mass* (self.a**2 + (self.t**2)/4)
        self.iyy = self.mass * ((self.a**2)/2 + (self.t**2)/8 + (self.l**2)/12)
        self.izz = self.iyy

class HybridMassModel:
    """Mass model for a a rocket that uses hybrid fuel

    Notes
    ----------
    - Assumes the solid fuel is an annular cylinder, and the liquid fuel is in a cylindrical fuel tank
    - Assumes the fuel tanks and solid fuel are coaxial along the rocket's long axis
    - The data for the solid fuel, liquid fuel, and vapour mass (v_mass) must all correspond to the same time array
    - Vapour mass data is only used for calculating the total mass. It is ignored for moments of inertia, and for the centre of gravity.
    WARNING: Might need to include the vapour mass for centre of gravity calculations for more accurate results?

    Parameters
    ----------

    Attributes
    ----------


    """
    def __init__(self, rocket_length, solid_fuel, liquid_fuel, vap_mass, dry_mass, dry_ixx, dry_iyy, dry_izz, dry_cog):
        self.solid_fuel = solid_fuel        #SolidFuel object
        self.liquid_fuel = liquid_fuel      #LiquidTank object
        self.time = self.liquid_fuel.time    

        #Vapour mass
        self.vap_mass_interp = scipy.interpolate.interp1d(self.time, vap_mass)

        #Properties without the fuel loaded
        self.dry_mass = dry_mass
        self.dry_cog = dry_cog              #centre of gravity (distance from the rocket's nose tip) when no fuel is loaded
        self.dry_ixx = dry_ixx
        self.dry_iyy = dry_iyy
        self.dry_izz = dry_izz

        #Geometry of the rocket
        self.l = rocket_length

    def vap_mass(self, time):
        #Mass of vapour
        if time<0:
            raise ValueError("Tried to input negative time when using HyrbidMassModel.vap_mass()")
        elif time < self.time[0]:
            return self.vap_mass_interp(self.time[0])
        elif time < self.time[-1]:
            return self.vap_mass_interp(time)
        else:
            return self.vap_mass_interp(self.time[-1])

    def mass(self, time):
        return self.dry_mass + self.vap_mass(time) + self.solid_fuel.mass(time) + self.liquid_fuel.mass(time)

    def cog(self, time):
        #Centre of gravity (distance from nose tip)
        return (self.dry_mass*self.dry_cog + self.liquid_fuel.mass(time)*self.liquid_fuel.cog(time) + self.solid_fuel.mass(time)*self.solid_fuel.cog(time)) / self.mass(time)
                                                                                    
    def ixx(self, time):
        #Parralel axis theorem (but in this case mR^2 = 0 for everything, because all cogs are on the xx axis)
        return self.dry_ixx + self.solid_fuel.ixx(time) + self.liquid_fuel.ixx(time)

    def iyy(self, time):
        #Get the parralel axis theorem equivelant of each mass, shifted to the rocket's current overall centre of gravity
        dry_component = self.dry_iyy + self.dry_mass*(self.cog(time) - self.dry_cog)**2
        liquid_component = self.liquid_fuel.iyy(time) + self.liquid_fuel.mass(time)*(self.cog(time) - self.liquid_fuel.cog(time))**2
        solid_component = self.solid_fuel.iyy(time) + self.solid_fuel.mass(time)*(self.cog(time) - self.solid_fuel.cog(time))**2

        return dry_component + liquid_component + solid_component

    def izz(self, time):
        #Get the parralel axis theorem equivelant of each mass, shifted to the rocket's current overall centre of gravity
        dry_component = self.dry_izz + self.dry_mass*(self.cog(time) - self.dry_cog)**2
        liquid_component = self.liquid_fuel.izz(time) + self.liquid_fuel.mass(time)*(self.cog(time) - self.liquid_fuel.cog(time))**2
        solid_component = self.solid_fuel.izz(time) + self.solid_fuel.mass(time)*(self.cog(time) - self.solid_fuel.cog(time))**2

        return dry_component + liquid_component + solid_component

    



