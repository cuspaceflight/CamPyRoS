'''
Liquid fuel slosh modelling tools

References
-----------
[1] - The Dynamic Behavior of Liquids in Moving Containers, with Applications to Space Vehicle Technology
Hyperlink - https://ntrs.nasa.gov/citations/19670006555

'''

import numpy as np

__copyright__ = """

    Copyright 2021 Daniel Gibbons

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

class CylindricalFuelTank:
    def __init__(self, h, d, rho):
        '''
        Cylindrical fuel tank
        
        Inputs
        -------
        h : float
            Liquid height (m)
        d : float
            Fuel tank diameter (m)
        rho : float
            Liquid density (kg m^-3)
        '''
        self.h = h
        self.d = d
        self.rho = rho
        
        #Table 6.1 from Reference [1]
        self.mt = 0.25 * np.pi * rho * h * d**2
        self.I_rigid = self.mt * d**2 * (1/12 * (h/d)**2 + 1/16)
    
    def pendulum_analogy(self):
        #Table 6.1 from Reference [1]
        #Let A = 3.68*h/d
        A = 3.68*self.h/self.d
        L1 = (self.d/3.68) * 1/np.tanh(A)
        m1 = self.mt * self.d /(4.4*self.h) * np.tanh(A)
        m0 = self.mt - m1
        l1 = -self.d/7.36 * 1/np.sinh(2*A)
        l0 = self.mt/m0 * (self.h/2 - self.d**2/(8*self.h)) - (l1 + L1)*(m1/m0)
        
        I0 = self.I_rigid + self.mt*(self.h/2)**2 - self.mt*self.d**2/8 * (1.995 - self.d/self.h * ((1.07*np.cosh(A) - 1.07)/np.sinh(A))) - m0*l0**2 - m1*(l1 + L1)
        
        return L1, m1, m0, l1, l0, I0
        
    def spring_analogy(self):
        #Table 6.1 from Reference [1]
        #Let A = 3.68*h/d
        A = 3.68*self.h/self.d
        K1 = self.mt*(9.81/(1.19*self.h))*np.tanh(A)**2
        m1 = self.mt * self.d /(4.4*self.h) * np.tanh(A)
        m0 = self.mt - m1
        l1 = (self.d/3.68)*np.tanh(A)
        l0 = self.mt/m0 * (self.h/2 - self.d**2/(8*self.h)) - l1*(m1/m0)
        
        I0 = self.I_rigid + self.mt*(self.h/2)**2 - self.mt*self.d**2/8 * (1.995 - self.d/self.h * ((1.07*np.cosh(A) - 1.07)/np.sinh(A))) - m0*l0**2 - m1*l1
        
        return K1, m1, m0, l1, l0, I0
    
    def w_spring(self):
        K1, m1, m0, l1, l0, I0 = self.spring_analogy()
        return (K1/m1)**0.5
    
    def w_pendulum(self):
        L1, m1, m0, l1, l0, I0 = self.pendulum_analogy()
        return (9.81/L1)**0.5
        
