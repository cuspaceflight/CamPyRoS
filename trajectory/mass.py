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
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
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
            raise ValueError("Tried to input negative time when using CylindricalMassModel.ixx()")
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