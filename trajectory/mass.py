"""
Notes
-----

- All "positions" are relative to the tip of the rocket's nose
- All "times" are from the moment of ignition

"""

import numpy as np

__copyright__ = """

    Copyright 2021 Jago Strong-Wright & Daniel Gibbons

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

"""

'''Mass models that have time-dependent properties'''    
class CylindricalApproximation:
    '''
    Solid cylinder approxiation for the rocket.

    Notes
    -----
    Assumes:
    - The entire rocket is a solid cylinder. It's volume is constant, but it's density decreases with time according to how the mass decreases as fuel is burnt.
    '''

    def __init__(self, mass_array, time_array, r, l):    
        self.mass_array = mass_array    #Mass of the rocket (kg)
        self.time_array = time_array    #Times since ignition corresponding to the mass datapoints (s)
        self.r = r                      #Cylinder radius (m)
        self.l = l                      #Cylinder length (m)

    def mass(self, time):  
        return np.interp(time, self.time_array, self.mass_array)

    def ixx(self, time): 
        return (1/2)* self.r**2 * self.mass(time)

    def iyy(self, time):
        return ((1/4)*self.r**2 + (1/12)*self.l**2) * self.mass(time)

    def izz(self, time):
        return self.iyy(time)
    
    def cog(self, time):
        return self.l/2

class LiquidTank:
    '''
    Liquid fuel tank.

    Notes
    -----
    Assumes:
    - Cylindrical fuel tank
    - Inviscid liquid, so the liquid does not contribute to ixx
    - Vapour does not contribute to moments of inertia
    '''

    def __init__(self, lmass_array, lden_array, time_array, r, pos_bottom, vmass_array = None, vden_array = None):  
        self.lmass_array = lmass_array          #Liquid masses (kg)
        self.lden_array = lden_array            #Liquid densities (kg/m^3)
        self.time_array = time_array            #Times since ignition for mass and density datapoints (s)

        self.r = r                              #Fuel tank radius (m)
        self.pos_bottom = pos_bottom  #Distance between nose tip and the bottom of the fuel tank

        self.vmass_array = vmass_array          #Vapour masses (kg)
        self.vden_array = vden_array            #Vapour densities (kg/m^3)

        #It's unclear what to do with the COG if the user puts in vapour mass data without vapour density data
        if vmass_array != None and vden_array == None:
            raise ValueError("You must input data for both vmass_array and vden_array. Without vden_array data, the position of the vapour COG is unclear")

    def lmass(self, time):
        return np.interp(time, self.time_array, self.lmass_array)

    def vmass(self, time):
        return np.interp(time, self.time_array, self.vmass_array)
    
    def lden(self, time):
        return np.interp(time, self.time_array, self.lden_array)

    def vden(self, time):
        return np.interp(time, self.time_array, self.vden_array)

    def lvol(self, time):        
        return self.lmass(time)/self.lden(time)
    
    def vvol(self, time):
        return self.vmass(time)/self.vden(time)

    def lheight(self, time):
        return self.lvol(time)/(np.pi*self.r**2)

    def vheight(self, time):
        return self.vvol(time)/(np.pi*self.r**2)

    def mass(self, time):
        if self.vmass_array == None:
            return self.lmass(time)
        else:
            return self.lmass(time) + self.vmass(time)

    def ixx(self, time):
        #Liquid is assumed to have no moment of inertia about the long axis (basically assumes it's inviscid so doesnt rotate with the rocket)
        return 0.0

    def iyy(self, time):
        #Assume a uniform solid cylinder
        return self.lmass(time) * (self.r**2 / 4 + self.lheight(time)**2 / 12)

    def izz(self, time):
        return self.iyy(time)

    def cog(self, time):
        lcog = self.pos_bottom - self.lheight(time)/2      #Liquid centre of gravity

        #If the user didn't input vapour data:
        if self.vmass_array == None:
            cog = lcog

        #If the user did input vapour data:
        if self.vmass_array != None:
            vcog = self.pos_bottom - self.lheight(time) - self.vheight(time)/2
            cog = (vcog*self.vmass(time) + lcog*self.lmass(time)) / self.mass(time)

        return cog

class SolidFuel:
    '''
    Solid fuel grain.

    Notes
    -----
    Assumes:
    - Fuel grain is shaped like an annular cylinder
    - Burning the fuel simply increases the inner radius of the cylinder, uniformly
    '''

    def __init__(self, mass_array, time_array, den, r_out, l, pos_bottom):  
        self.mass_array = mass_array    #Solid fuel masses (kg)
        self.time_array = time_array    #Times since ignition for mass datapoints (s)

        self.den = den                  #Density of solid fuel grain (kg/m^3)
        self.r_out = r_out              #Outer radius of solid fuel (kg/m^3)
        self.l = l                      #Length of solid fuel grain (kg/m^3)
        self.pos_bottom = pos_bottom    #Distance between the bottom of the fuel grain and the rocket nose tip
    
    def cog(self, time):
        return self.pos_bottom - self.l/2  

    def mass(self, time):
        return np.interp(time, self.time_array, self.mass_array)
    
    def r_in(self, time):
        return (self.r_out**2 - self.mass(time)/(np.pi * self.den * self.l))**0.5

    def ixx(self, time):
        a = (self.r_out + self.r_in(time))/2    #Average radius
        t = self.r_out - self.r_in(time)        #Thickness

        return self.mass(time) * (a**2 + (t**2)/4)

    def iyy(self, time):
        a = (self.r_out + self.r_in(time))/2    #Average radius
        t = self.r_out - self.r_in(time)        #Thickness

        return self.mass(time) * ((a**2)/2 + (t**2)/8 + (self.l**2)/12)

    def izz(self, time):
        return self.iyy(time)


'''Mass models that have constant properties'''
class DryMass:
    '''
    Class for adding custom dry mass values.
    '''

    def __init__(self, mass, ixx, iyy, izz, cog):
        self.mass = mass
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz
        self.cog = cog
        
class HollowCylinder:
    '''
    Class for getting moment of inertia properties for a solid hollow cylinder that does not vary with time.
    '''

    def __init__(self, mass, r_out, r_in, l, cog):
        self.r_out = r_out  #Outer radius (m)
        self.r_in = r_in    #Inner radius (m)
        self.l = l          #Length (m)

        self.t = self.r_out - self.r_in         #Wall thickness (m)
        self.a = (self.r_out + self.r_in)/2     #Average radius (m)

        self.mass = mass
        self.cog = cog
        self.ixx = self.mass * (self.a**2 + (self.t**2)/4)
        self.iyy = self.mass * ((self.a**2)/2 + (self.t**2)/8 + (self.l**2)/12)
        self.izz = self.iyy

'''Universal mass model'''
class MassModel:
    '''
    Notes
    -----
    Assumes:
    - All centres of mass lie on the x-x axis.
    '''

    def __init__(self):
        self.constants = []
        self.variables = []

    '''Mass-related properties of the rocket'''
    def mass(self, time):
        mass = 0

        for mass_model in self.constants:
            mass = mass + mass_model.mass

        for mass_model in self.variables:
            mass = mass + mass_model.mass(time)

        return mass

    def cog(self, time):
        cog = 0

        for mass_model in self.constants:
            cog = cog + mass_model.mass * mass_model.cog

        for mass_model in self.variables:
            cog = cog + mass_model.mass(time) * mass_model.cog(time)

        cog = cog/self.mass(time)

        return cog

    def ixx(self, time):
        ixx = 0
        cog = self.cog(time)

        #Parralel axis theorem is not necessary for ixx since the body is axially symmetric about x-x.
        for mass_model in self.constants:
            ixx = ixx + mass_model.ixx

        for mass_model in self.variables:
            ixx = ixx + mass_model.ixx(time)

        return ixx
    
    def iyy(self, time):
        iyy = 0

        #Parralel axis theorem
        for mass_model in self.constants:
            iyy = iyy + mass_model.iyy + mass_model.mass * (mass_model.cog - self.cog(time))**2

        for mass_model in self.variables:
            iyy = iyy + mass_model.iyy(time) + mass_model.mass(time) * (mass_model.cog(time) - self.cog(time))**2

        return iyy

    def izz(self, time):
        izz = 0

        #Parralel axis theorem
        for mass_model in self.constants:
            izz = izz + mass_model.izz + mass_model.mass * (mass_model.cog - self.cog(time))**2

        for mass_model in self.variables:
            izz = izz + mass_model.izz(time) + mass_model.mass(time) * (mass_model.cog(time) - self.cog(time))**2

        return izz


    '''Functions to add new components'''
    def add_drymass(self, mass, ixx, iyy, izz, cog):
        self.constants.append(DryMass(mass, ixx, iyy, izz, cog))

    def add_liquidtank(self, lmass_array, lden_array, time_array, r, pos_bottom, vmass_array = None, vden_array = None):
        self.variables.append(LiquidTank(lmass_array, lden_array, time_array, r, pos_bottom, vmass_array = None, vden_array = None))

    def add_solidfuel(self, mass_array, time_array, den, r_out, l, pos_bottom):
        self.variables.append(SolidFuel(mass_array, time_array, den, r_out, l, pos_bottom))

    def add_cylindricalapproximation(self, mass_array, time_array, r, l): 
        self.variables.append(CylindricalApproximation(mass_array, time_array, r, l))

    def add_hollowcylinder(self, mass, r_out, r_in, l, cog):
        self.constants.append(HollowCylinder(mass, r_out, r_in, l, cog))
