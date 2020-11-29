import numpy as np
from scipy.spatial.transform import Rotation

from trajectory.constants import r_earth, ang_vel_earth

#pos_l2i and pos_i2l HAVE BEEN CHANGED BUT HAS NOT BEEN TESTED        
def pos_l2i(pos_l, launch_site, time):
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -Converting spherical coordinates to Cartesian
    -https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2)

    Parameters
    ----------
    pos_l : numpy array
        Position in the launch site frame [x,y,z] /m
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Position in the inertial frame
    """
    
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    pos_rocket_i = pos_launch_site_i + direction_l2i(pos_l, launch_site, time)

    return pos_rocket_i

def pos_i2l(position,launch_site,time):
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -Converting spherical coordinates to Cartesian
    -https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2)

    Parameters
    ----------
    position : numpy array
        Position in the inertial frame [x,y,z] /m
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Position in the launch frame
    """
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    pos_rocket_i = position
    pos_rocket_l =  direction_i2l(pos_rocket_i - pos_launch_site_i, launch_site, time)

    return pos_rocket_l

def vel_i2l(vel_i, launch_site, time): 
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -v = w x r for a rigid body, where v, w and r are vectors

    Parameters
    ----------
    vel_i : numpy array
        Velocity in the inertial frame [x,y,z] /m/s
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Velocity in the launch frame
    """  
    w_earth = np.array([0, 0, ang_vel_earth])
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                    r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                    r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    launch_site_velocity_i = np.cross(w_earth, pos_launch_site_i)

    return direction_i2l(vel_i - launch_site_velocity_i, launch_site, time) 

def vel_l2i(vel_l, launch_site, time):
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -v = w x r for a rigid body, where v, w and r are vectors

    Parameters
    ----------
    vel_i : numpy array
        Velocity in the launch frame [x,y,z] /m/s
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Velocity in the inertial frame
    """ 
    w_earth = np.array([0, 0, ang_vel_earth])
    pos_launch_site_i = [r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.cos(launch_site.longi * np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.sin((90 - launch_site.lat) * np.pi / 180.0) * np.sin(launch_site.longi* np.pi / 180.0 + ang_vel_earth*time),
                        r_earth * np.cos((90 - launch_site.lat) * np.pi / 180.0)]

    launch_site_velocity_i = np.cross(w_earth, pos_launch_site_i)

    return direction_l2i(vel_l, launch_site, time) + launch_site_velocity_i

def direction_i2l(vector, launch_site, time):
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -Problem in the yaw pitch conversions, unexplained negative sign needed

    Parameters
    ----------
    vector : numpy array
        Vector in the inertial frame [x,y,z] /m/s
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Vector in the launch frame
    """ 
    return Rotation.from_euler('zy', [-launch_site.longi - (180/np.pi)*ang_vel_earth*time, -90 + launch_site.lat], degrees=True).apply(vector)

def direction_l2i(vector, launch_site, time):
    """Converts position in launch frame to position in inertial frame.

    Note
    ----
    -Problem in the yaw pitch conversions, unexplained negative sign needed

    Parameters
    ----------
    vector : numpy array
        Vector in the launch frame [x,y,z] /m/s
    launch_site : LaunchSite object
        Holds the launch site parameters
    time : float
        Time since ignition /s

    Returns
    -------
    numpy array
        Vector in the launch frame
    """ 
    return Rotation.from_euler('zy', [-launch_site.longi - (180/np.pi)*ang_vel_earth*time, -90 + launch_site.lat], degrees=True).inv().apply(vector)
