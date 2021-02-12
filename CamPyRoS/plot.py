"""6DOF Trajectory Simulator

Various useful plots of the outputted data

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from .transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, pos_i2alt, i2airspeed,i2lla
import pandas as pd
from scipy.spatial.transform import Rotation
from ambiance import Atmosphere

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

def get_velocity_magnitude(df):
    return (np.sqrt(df["vx_l"]**2+df["vy_l"]**2+df["vz_l"]**2))

#Functional
def set_axes_equal_3d(ax):
    """
    Makes the scaling the same on the axes of 3D plot. The in-built functions that come with matplotlib only seem to be able to do this for 2D axes.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes
        The 3D axis that you want to have equal axis scaling, e.g. could have been created with ax = matplotlib.pyplot.axes(projection="3d").

    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    #Get info on the data that has already been plotted on the axes
    x_range = abs(x_limits[1] - x_limits[0])    #Range of the x axis data.
    x_middle = (x_limits[1] + x_limits[0])/2    #The midpoint of the x axis data.
    
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = (y_limits[1] + y_limits[0])/2
    
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = (z_limits[1] + z_limits[0])/2
    
    #Get the maximum range present of all 3 axes. We will use this to determine the size of the plotting area that we want.
    max_range = max([x_range, y_range, z_range])
    
    #Set each axis limit so the range present on each axis is the same, and is equal to the max range of all the data out of the three axes.
    ax.set_xlim3d([x_middle - 0.5*max_range, x_middle + 0.5*max_range])
    ax.set_ylim3d([y_middle - 0.5*max_range, y_middle + 0.5*max_range])
    ax.set_zlim3d([z_middle - 0.5*max_range, z_middle + 0.5*max_range])


def fix_ypr(point):
    return point

def plot_ypr(simulation_output, rocket):
    #Get data
    output_dict = simulation_output.to_dict(orient="list")
    
    yaw=[]
    pitch=[]
    roll=[]
    z_l=[]
    for index, row in simulation_output.iterrows():#this is ugly but proper way not working
        ypr=Rotation.from_matrix(row["b2imat"]).as_euler("zyx")
    altitude=[]
    time = output_dict["time"]

    for i in range(len(time)):
        ypr = Rotation.from_matrix(output_dict["b2imat"][i]).as_euler("zyx")
        yaw.append(ypr[0])
        pitch.append(ypr[1])
        roll.append(ypr[2])
        z_l.append(pos_i2l(np.array(row["pos_i"]),rocket.launch_site,row["time"])[2])
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(simulation_output["time"], [fix_ypr(n) for n in yaw])
    axs[0, 0].set_title('Yaw')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Angles/ rad")

    axs[0, 1].plot(simulation_output["time"], [fix_ypr(n) for n in pitch])
    axs[0, 1].set_title('Pitch')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Angles/ rad")

    axs[1, 0].plot(simulation_output["time"], [fix_ypr(n) for n in roll])
    axs[1, 0].set_title('Roll')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Angles/ rad")

    axs[1, 1].plot(simulation_output["time"], z_l)
    axs[1, 1].set_title('Altitude')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Altitude /m")

    plt.show()

def plot_launch_trajectory_3d(simulation_output, rocket, show_orientation=False, arrow_frequency = 0.02):
    """
    Plots the trajectory in 3D, given the simulation_output and the rocket

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:

    rocket : trajectory.Rocket object
        The rocket object that was used to produce the simulation data. Is needed to calculate coordinate system changes.

    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    #Convert inertial positions to launch site ones
    pos_i_array = np.stack(simulation_output["pos_i"], axis=0)    #np.stack(Series, axis=0) will convert a panda Series into a numpy ndarray. 
    pos_l_array = np.zeros(pos_i_array.shape)

    for i in range(len(pos_l_array)):
        pos_l_array[i] = pos_i2l(pos_i_array[i], rocket.launch_site, simulation_output["time"][i])
    x_l = pos_l_array[:, 0]
    y_l = pos_l_array[:, 1]
    z_l = pos_l_array[:, 2]
    
    #Plot rocket positions
    ax.plot3D(x_l, y_l, z_l)
    ax.set_xlabel('South')
    ax.set_ylabel('East')
    ax.set_zlabel('Altitude')  

    #Plot launch site position and burnout position
    ax.scatter(x_l[0], y_l[0], z_l[0], c='red', label="Launch site", linewidths=3)
    
    #Get burnout position
    burnout_time = rocket.motor.time_array[-1]
    burnout_index = (np.abs(np.array(simulation_output["time"]) - burnout_time)).argmin()
    ax.scatter(x_l[burnout_index], y_l[burnout_index], z_l[burnout_index], c='green', label="Engine burnout", linewidths=3)

    
    #Indexes to plot arrows at
    idx = np.linspace(0, len(x_l) - 1, int(arrow_frequency*len(x_l))).astype(int)
    
    #Plot the direction the rocket faces at each point (i.e. direction of xb_l), using quivers
    if show_orientation==True:
        b2imat = np.stack(simulation_output["b2imat"], axis=0)
        xb_i_array = b2imat[:, :, 0]

        #Convert inertial orientations into launch site ones
        xb_l_array = np.zeros(xb_i_array.shape)
        for i in range(len(xb_i_array)):
            xb_l_array[i] = direction_i2l(xb_i_array[i], rocket.launch_site, simulation_output["time"][i])

        u = xb_l_array[:, 0]
        v = xb_l_array[:, 1]
        w = xb_l_array[:, 2]

        ax.quiver(x_l[idx], y_l[idx], z_l[idx], u[idx], v[idx], w[idx], length=1000, normalize=True, color="red", label="Orientation")
 
    #Make all the axes scales equal
    set_axes_equal_3d(ax)
    
    ax.legend()
    plt.show()     

def plot_altitude_time(simulation_output, rocket):
    """
    Plots the following, against time where applicable: ground track, altitude, speed (in the launch frame) and vertical velocity (in the launch frame)

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:

    rocket : trajectory.Rocket object
        The rocket object that was used to produce the simulation data. Is needed to calculate coordinate system changes.


    """
    #Convert inertial positions to launch site ones
    pos_i_array = np.stack(simulation_output["pos_i"], axis=0)     #np.stack(Series, axis=0) will convert a panda Series into a numpy ndarray
    pos_l_array = np.zeros(pos_i_array.shape)

    for i in range(len(pos_l_array)):
        pos_l_array[i] = pos_i2l(pos_i_array[i], rocket.launch_site, simulation_output["time"][i])

    x_l = pos_l_array[:, 0]
    y_l = pos_l_array[:, 1]
    z_l = pos_l_array[:, 2]

    #Convert inertial velocities into launch site ones
    vel_i_array = np.stack(simulation_output["vel_i"], axis=0)     #np.stack(Series, axis=0) will convert a panda Series into a numpy ndarray
    vel_l_array = np.zeros(pos_i_array.shape)

    for i in range(len(pos_l_array)):
        vel_l_array[i] = vel_i2l(vel_i_array[i], rocket.launch_site, simulation_output["time"][i])

    vx_l = vel_l_array[:, 0]
    vy_l = vel_l_array[:, 1]
    vz_l = vel_l_array[:, 2]
    
    #Plot everything
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(y_l, -x_l)
    axs[0, 0].set_title('Ground Track ($^*$)')
    axs[0,0].set_xlabel("East/m")
    axs[0,0].set_ylabel("North/m")

    axs[0, 1].plot(simulation_output["time"], z_l, 'tab:orange')
    axs[0, 1].set_title('Altitude')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Altitude/m")

    axs[1, 0].plot(simulation_output["time"], (vx_l**2 + vy_l**2 + vz_l**2)**0.5, 'tab:green')
    axs[1, 0].set_title('Speed')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Speed/m/s")

    axs[1, 1].plot(simulation_output["time"], vz_l, 'tab:red')
    axs[1, 1].set_title('Vertical Velocity')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Velocity/m/s")

    fig.tight_layout()
    plt.show() 

def plot_aero(simulation_output, rocket):
    """
    Plots the following:
    - Aerodynamic forces against time
    - COG and COP against time
    - Angles of attack against time

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:

    rocket : trajectory.Rocket object
        The rocket object that was used to produce the simulation data. Is needed to calculate coordinate system changes.
    """
  
    output_dict = simulation_output.to_dict(orient="list")
    burnout_time = rocket.motor.motor_time_data[-1]

    force_x_b = []
    force_y_b = []
    force_z_b = []

    moment_x_b = []
    moment_y_b = []
    moment_z_b = []

    q_data = []
    cop_data = []
    cog_data = []
    alpha_data = []

    print("Generating data for aerodynamic plot - may take a while")
    message = False
    for i in range(len(output_dict["time"])):
        b2i = Rotation.from_matrix(output_dict["b2imat"][i])

        force_b, cop, q = rocket.aero_forces(output_dict["pos_i"][i], output_dict["vel_i"][i], b2i, output_dict["w_b"][i], output_dict["time"][i])

        force_x_b.append(force_b[0])
        force_y_b.append(force_b[1])
        force_z_b.append(force_b[2])

        damping_moment = rocket.aero_damping_moment(output_dict["pos_i"][i], output_dict["vel_i"][i], b2i, output_dict["w_b"][i], output_dict["time"][i])

        moment_x_b.append(damping_moment[0])
        moment_y_b.append(damping_moment[1])
        moment_z_b.append(damping_moment[2])

        q_data.append(q)
        cop_data.append(-cop)
        cog_data.append(-rocket.mass_model.cog(output_dict["time"][i]))
        lat,long,alt=i2lla(output_dict["pos_i"][i],output_dict["time"][i])
        v_rel_wind = b2i.inv().apply( direction_l2i((i2airspeed(output_dict["pos_i"][i], output_dict["vel_i"][i], rocket.launch_site, output_dict["time"][i]) - rocket.launch_site.wind.get_wind(lat,long,alt)), rocket.launch_site,  output_dict["time"][i]) )
        airspeed = np.linalg.norm(v_rel_wind)
        alpha_data.append(np.arccos(np.dot(v_rel_wind/airspeed, [1,0,0])))

        if message==False and i>len(output_dict["time"])/2:
            print("50% complete")
            message = True
    
    
    print("Finished")

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('Aerodynamic Force')
    axs[0, 0].plot(output_dict["time"], force_x_b, label="X-force (body-coordinates)", color="red")
    axs[0, 0].plot(output_dict["time"], force_y_b, label="Y-force (body-coordinates)", color="green")
    axs[0, 0].plot(output_dict["time"], force_z_b, label = "Z-force (body-coordinates)", color="blue")
    axs[0, 0].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[0,0].set_xlabel("Time (s)")
    axs[0,0].set_ylabel("Force (N)")
    axs[0,0].legend()
    axs[0,0].grid()

    axs[0, 1].set_title('Aerodynamic Damping Moment')
    axs[0, 1].plot(output_dict["time"], moment_x_b, label="X-moment (body-coordinates)", color="red")
    axs[0, 1].plot(output_dict["time"], moment_y_b, label="Y-moment (body-coordinates)", color="green")
    axs[0, 1].plot(output_dict["time"], moment_z_b, label = "Z-moment (body-coordinates)", color="blue")
    axs[0, 1].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[0,1].set_xlabel("Time (s)")
    axs[0,1].set_ylabel("Moment (Nm)")
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,0].set_title('Dynamic pressure')
    axs[1,0].plot(output_dict["time"], np.array(q_data)/1000, label="Dynamic pressure")
    axs[1,0].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[1,0].set_xlabel("Time (s)")
    axs[1,0].set_ylabel("Dynamic Pressure (kPa)")
    axs[1,0].legend()
    axs[1,0].grid()

    axs[1,1].set_title('Angle of attack, centre of pressure (COP), and centre of gravity (COG)')
    axs[1,1].plot(output_dict["time"], np.array(alpha_data) * 180/np.pi, color="red", label="Angle of attack")

    cop_axis = axs[1,1].twinx() #Plot COP and COG on the same graph as the angle of attack, but use a different y-scale
    cop_axis.plot(output_dict["time"], cop_data, label="COP", color="green")
    cop_axis.plot(output_dict["time"], cog_data, label="COG", color="blue")
    cop_axis.set_ylabel("COP and COG position (m)")
    cop_axis.legend()

    axs[1,1].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[1,1].set_xlabel("Time (s)")
    axs[1,1].set_ylabel("Angle of attack (deg)")
    axs[1,1].legend()
    axs[1,1].grid()

    plt.show()

def plot_thrust(simulation_output, rocket):
    """
    Plots the following:
    - Thrust against time
    - Jet damping moment against time
    - Propellant mass flow rate against time

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:

    rocket : trajectory.Rocket object
        The rocket object that was used to produce the simulation data. Is needed to calculate coordinate system changes.
    """
    
    output_dict = simulation_output.to_dict(orient="list")
    burnout_time = rocket.motor.motor_time_data[-1]

    thrust_x_b = []
    thrust_y_b = []
    thrust_z_b = []

    jet_damping_x_b = []
    jet_damping_y_b = []
    jet_damping_z_b = []

    mdot_data = []

    for i in range(len(output_dict["time"])):
        thrust_b, jet_damping_moment = rocket.thrust(output_dict["pos_i"][i], output_dict["vel_i"][i], Rotation.from_matrix(output_dict["b2imat"][i]), output_dict["w_b"][i], output_dict["time"][i])
        thrust_x_b.append(thrust_b[0])
        thrust_y_b.append(thrust_b[1])
        thrust_z_b.append(thrust_b[2])

        jet_damping_x_b.append(jet_damping_moment[0])
        jet_damping_y_b.append(jet_damping_moment[1])
        jet_damping_z_b.append(jet_damping_moment[2])

        if output_dict["time"][i] < max(rocket.motor.motor_time_data):
            mdot = np.interp(output_dict["time"][i], rocket.motor.motor_time_data, rocket.motor.mdot_data)
        else:
            mdot = 0
        mdot_data.append(mdot)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(output_dict["time"], thrust_x_b, label="X-force (body-coordinates)", color="red")
    axs[0, 0].plot(output_dict["time"], thrust_y_b, label="Y-force (body-coordinates)", color="green")
    axs[0, 0].plot(output_dict["time"], thrust_z_b, label = "Z-force (body-coordinates)", color="blue")
    axs[0, 0].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[0, 0].set_title('Thrust Force')
    axs[0,0].set_xlabel("Time (s)")
    axs[0,0].set_ylabel("Force (N)")
    axs[0,0].legend()
    axs[0,0].grid()
    
    axs[0, 1].plot(output_dict["time"], jet_damping_x_b, label="X-moment (body-coordinates)", color="red")
    axs[0, 1].plot(output_dict["time"], jet_damping_y_b, label="Y-moment (body-coordinates)", color="green")
    axs[0, 1].plot(output_dict["time"], jet_damping_z_b, label = "Z-moment (body-coordinates)", color="blue")
    axs[0, 1].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[0, 1].set_title('Jet Damping Moment')
    axs[0,1].set_xlabel("Time (s)")
    axs[0,1].set_ylabel("Moment (Nm)")
    axs[0,1].legend()
    axs[0,1].grid()
    
    axs[1,0].plot(output_dict["time"], mdot_data)
    axs[1, 0].axvline(burnout_time, label="Burnout time", linestyle = '--', color="black")
    axs[1,0].set_title('Exhaust mass flow rate')
    axs[1,0].set_xlabel("Time (s)")
    axs[1,0].set_ylabel("Mass flow rate (kg/s)")
    axs[1,0].grid()
    
    plt.show()
        
def plot_mass(simulation_output, rocket):
    """
    Plots the following:
    - Total mass against time
    - Moments of inertia against time
    - Angles of attack against time

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:

    rocket : trajectory.Rocket object
        The rocket object that was used to produce the simulation data. Is needed to calculate coordinate system changes.
    """
  
    time = simulation_output.to_dict(orient="list")["time"]
    burnout_time = rocket.motor.motor_time_data[-1]
    mass = []
    ixx = []
    iyy = []
    izz = []
    for t in time:
        mass.append(rocket.mass_model.mass(t))
        ixx.append(rocket.mass_model.ixx(t))
        iyy.append(rocket.mass_model.iyy(t))
        izz.append(rocket.mass_model.izz(t))


    #Plot everything
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(time, mass)
    axs[0, 0].set_title('Mass')
    axs[0,0].set_xlabel("Time (s)")
    axs[0,0].set_ylabel("Mass (kg)")
    axs[0,0].axvline(burnout_time, label="Burnout time", linestyle = '--')
    axs[0,0].legend()
    axs[0,0].grid()

    axs[0, 1].plot(time, ixx, 'tab:orange')
    axs[0, 1].set_title('Ixx')
    axs[0,1].set_xlabel("time (s)")
    axs[0,1].set_ylabel("Ixx (kg m^2)")
    axs[0,1].axvline(burnout_time, label="Burnout time", linestyle = '--', color="orange")
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1, 0].plot(time, iyy, 'tab:green')
    axs[1, 0].set_title('Iyy')
    axs[1,0].set_xlabel("time (s)")
    axs[1,0].set_ylabel("Iyy (kg m^2)")
    axs[1,0].axvline(burnout_time, label="Burnout time", linestyle = '--', color="green")
    axs[1,0].legend()
    axs[1,0].grid()

    axs[1, 1].plot(time, izz, 'tab:red')
    axs[1, 1].set_title('Izz')
    axs[1,1].set_xlabel("time (s)")
    axs[1,1].set_ylabel("Izz (kg m^2)")
    axs[1,1].axvline(burnout_time, label="Burnout time", linestyle = '--', color="red")
    axs[1,1].legend()
    axs[1,1].grid()

    fig.tight_layout()
    plt.show() 
    
def animate_orientation(simulation_output, frames=500):
    """
    Shows an animation of the orientation against time, alongside an animation of altitude against time for reference

    Parameters
    ----------
    simulation_output: pandas array
        Simulation output from a Rocket.run() method. Should contain the following data:
    """
    
    output_dict = simulation_output.to_dict(orient="list")

    #Get data
    yaw=[]
    pitch=[]
    roll=[]
    altitude=[]
    time = output_dict["time"]

    for i in range(len(time)):
        ypr = Rotation.from_matrix(output_dict["b2imat"][i]).as_euler("zyx")
        yaw.append(ypr[0])
        pitch.append(ypr[1])
        roll.append(ypr[2])
        altitude.append(pos_i2alt(output_dict["pos_i"][i],output_dict["time"]))

    #Create figure
    fig, axs = plt.subplots(2, 2)
    
    #Add titles
    axs[0, 0].set_title('Yaw')
    axs[0, 1].set_title('Pitch')
    axs[1, 0].set_title('Roll')
    axs[1, 1].set_title('Altitude (m)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()    
    
    #Plot the initial directions
    axs[0,0].plot(np.linspace(0,np.cos(yaw[0]), 100), np.linspace(0, np.sin(yaw[0]), 100), lw=3, color='black')
    axs[0,1].plot(np.linspace(0,np.cos(pitch[0]), 100), np.linspace(0, np.sin(pitch[0]), 100), lw=3, color='black')
    axs[1,0].plot(np.linspace(0,np.cos(roll[0]), 100), np.linspace(0, np.sin(roll[0]), 100), lw=3, color='black')
    
    #Set up the lines that will be animated
    line1, = axs[0,0].plot([], [], lw=3, color='red')
    line2, = axs[0,1].plot([], [], lw=3, color='green')
    line3, = axs[1,0].plot([], [], lw=3, color='blue')
    line4, = axs[1,1].plot([], [], lw=3, color='orange')
    
    axs[0,0].set_xlim(-1,1)
    axs[0,0].set_ylim(-1,1)
    
    axs[0,1].set_xlim(-1,1)
    axs[0,1].set_ylim(-1,1)
    
    axs[1,0].set_xlim(-1,1)
    axs[1,0].set_ylim(-1,1)
    
    axs[1,1].set_xlim(0,max(time))
    axs[1,1].set_ylim(0,30e3)
    
    def init1():
        line1.set_data([], [])
        return line1,
    
    def init2():
        line2.set_data([], [])
        return line1,
    
    def init3():
        line3.set_data([], [])
        return line1,
    
    def init4():
        line4.set_data([], [])
        return line1,
    
    def update1(i):
        j = int(i*len(yaw)/frames)
        x = np.linspace(0,np.cos(yaw[j]), 100)
        y = np.linspace(0, np.sin(yaw[j]), 100)
        line1.set_data(x, y)
        return line1,
    
    def update2(i):
        j = int(i*len(pitch)/frames)
        x = np.linspace(0,np.cos(pitch[j]), 100)
        y = np.linspace(0, np.sin(pitch[j]), 100)
        line2.set_data(x, y)
        return line2,
    
    def update3(i):
        j = int(i*len(roll)/frames)
        x = np.linspace(0,np.cos(roll[j]), 100)
        y = np.linspace(0, np.sin(roll[j]), 100)
        line3.set_data(x, y)
        return line3,
    
    def update4(i):
        j = int(i*len(altitude)/frames)
        y = altitude[0:j]
        x = time[0:j]
        line4.set_data(x, y)
        return line4,
    
    #Start the animations
    anim1 = FuncAnimation(fig, update1, init_func=init1,
                                   frames=frames, blit=True)
    anim2 = FuncAnimation(fig, update2, init_func=init2,
                                   frames=frames, interval=20, blit=True)
    anim3 = FuncAnimation(fig, update3, init_func=init3,
                                   frames=frames, interval=20, blit=True)
    anim4 = FuncAnimation(fig, update4, init_func=init4,
                                   frames=frames, interval=20, blit=True)

    #The animation only seems to run if you trigger an error to do with axs, I don't know why:
    axs.i_dont_know_why_this_makes_it_work

    plt.show()

def stats_landing(mu,cov,data=pd.DataFrame(),sigma=3):
    t=np.linspace(0,2*np.pi,314)
    fig=plt.figure()
    ax=fig.gca()
    eig=np.linalg.eig(cov)
    eig_mat=np.array([[eig[1][0][1],eig[1][1][1]],[eig[1][0][0],eig[1][1][0]]])
    cov_elipse = np.matmul(eig_mat,np.array([eig[0][1]**0.5*np.cos(t),eig[0][0]**0.5*np.sin(t)]))
    
    for sig in range(1,sigma+1):
        ax.plot(sig*cov_elipse[0]+mu[0],sig*cov_elipse[1]+mu[1],label="%s $\sigma$"%sig,linewidth=1)

    ax.scatter(0,0,marker="x",color="red",label="Launch site")
    ax.scatter(mu[0],mu[1],marker="o",s=20,color="black",label="Mean landing point")

    ax.set_xlabel("South/m")
    ax.set_ylabel("East/m")

    if not data.empty:
        ax.scatter(data.x,data.y,marker="o",s=10,color="blue",alpha=0.3)
    ax.legend()
    plt.show()

def elipse(u,v,a,b,c):
    x=a*np.cos(u)*np.sin(v)
    y=b*np.sin(u)*np.sin(v)
    z=c*np.cos(v)
    return np.array([x,y,z])

def stats_apogee(apogee_mu,apogee_cov,apogee=pd.DataFrame(),sigma=3,landing_mu=np.array([]),landing_cov=np.array([]),landing=pd.DataFrame()):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    ap_eig=np.linalg.eig(apogee_cov)
    ap_eig_mat=np.array([[ap_eig[1][0][0],ap_eig[1][1][0],ap_eig[1][2][0]],[ap_eig[1][0][1],ap_eig[1][1][1],ap_eig[1][2][1]],[ap_eig[1][0][2],ap_eig[1][1][2],ap_eig[1][2][2]]])


    x,y,z=[],[],[]

    for v in np.linspace(0,np.pi,100):
        for u in np.linspace(0,2*np.pi,200):
            d=elipse(u,v,ap_eig[0][0]**.5,ap_eig[0][1]**.5,ap_eig[0][2]**.5)
            d=np.matmul(ap_eig_mat.transpose(),d)
            x.append(d[0])
            y.append(d[1])
            z.append(d[2])

    for sig in range(1,sigma+1):
        ax.plot(apogee_mu[0]+sig*np.array(x),apogee_mu[1]+sig*np.array(y),apogee_mu[2]+sig*np.array(z),alpha=.2)

    ax.plot(apogee_mu[0],apogee_mu[1],apogee_mu[2],label="Mean apogee point")

    ax.scatter(0,0,0,marker="x",color="red",label="Launch site")
    ax.set_xlabel("South/m")
    ax.set_ylabel("East/m")
    ax.set_zlabel("Altitude/m")

    if not apogee.empty:
        ax.scatter(apogee.x,apogee.y,apogee.alt,marker="o",s=10,color="blue",alpha=0.3)

    if landing_mu.shape!=(0,) and landing_cov.shape!=(0,):
        t=np.linspace(0,2*np.pi,314)

        l_eig=np.linalg.eig(landing_cov)
        l_eig_mat=np.array([[l_eig[1][0][0],l_eig[1][1][0]],[l_eig[1][0][1],l_eig[1][1][1]]])
        landing_cov_elipse = np.matmul(l_eig_mat.transpose(),np.array([l_eig[0][0]**0.5*np.cos(t),l_eig[0][1]**0.5*np.sin(t)]))
        
        for sig in range(1,sigma+1):
            ax.plot(sig*landing_cov_elipse[0]+landing_mu[0],sig*landing_cov_elipse[1]+landing_mu[1],0,label="%s $\sigma$"%sig,linewidth=1)

        ax.scatter(landing_mu[0],landing_mu[1],0,marker="o",s=20,color="black",label="Mean landing point")

        if not landing.empty:
            ax.scatter(landing.x,landing.y,0,marker="o",s=10,color="blue",alpha=0.3)


    set_axes_equal_3d(ax)
    ax.legend()
    plt.show()

def stats_alt(z,t,show_means=False,sigma=3):
    landing_times=[]
    for (itt, data) in z.iteritems():
        plt.plot(t[itt],data,linewidth=1,color="blue",alpha=0.2)
        landing_times.append(t[itt][z[itt].notna()[::-1].idxmax()])
    if show_means==True:
        mean_alts=[10]
        st_dev=[0]
        time=0
        incriments=1
        t_alts=[10]
        while max(t_alts)>0:
            t_alts=[]
            for (itt,data) in z.iteritems():
                alt=data[(t[itt]-time).abs().argsort()[:1]].astype(float).to_numpy()[0]
                t_alts.append(alt)
            mean_alts.append(np.mean(t_alts))
            st_dev.append(np.std(t_alts))
            time+=incriments
        mean_alts=mean_alts[1:]
        st_dev=st_dev[1:]
        plt.plot([incriments*n for n in range(0,len(mean_alts))],mean_alts,color="red",label="Mean")
        for sig in range(1,sigma+1):
            plt.plot([incriments*n for n in range(0,len(mean_alts))],np.array(mean_alts)+np.array(st_dev)*sig,alpha=(.9-.1*sig),color="orange",label="%s$\sigma$"%sig,linewidth=1)
            plt.plot([incriments*n for n in range(0,len(mean_alts))],np.array(mean_alts)-np.array(st_dev)*sig,alpha=(.9-.1*sig),color="orange",linewidth=1)
    plt.xlabel("Time/s")
    plt.ylabel("Altitude/m")
    plt.legend()
    plt.ylim(0,None)
    plt.show()
def stats_trajectories(x,y,z,apogee_mu=np.array([]),apogee_cov=np.array([]),sigma=3,landing_mu=np.array([]),landing_cov=np.array([])):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    for (itt,data) in z.iteritems():
        ax.plot3D(x[itt],y[itt],data,color="blue",alpha=0.3)
    
    ax.scatter(0,0,0,c='red', label="Launch site")

    if apogee_mu.shape!=(0,) and apogee_cov.shape!=(0,) and landing_mu.shape!=(0,) and landing_cov.shape!=(0,): 
        ap_eig=np.linalg.eig(apogee_cov)
        ap_eig_mat=np.array([[ap_eig[1][0][0],ap_eig[1][1][0],ap_eig[1][2][0]],[ap_eig[1][0][1],ap_eig[1][1][1],ap_eig[1][2][1]],[ap_eig[1][0][2],ap_eig[1][1][2],ap_eig[1][2][2]]])


        x,y,z=[],[],[]

        for v in np.linspace(0,np.pi,100):
            for u in np.linspace(0,2*np.pi,200):
                d=elipse(u,v,ap_eig[0][0]**.5,ap_eig[0][1]**.5,ap_eig[0][2]**.5)
                d=np.matmul(ap_eig_mat.transpose(),d)
                x.append(d[0])
                y.append(d[1])
                z.append(d[2])

        for sig in range(1,sigma+1):
            ax.plot(apogee_mu[0]+sig*np.array(x),apogee_mu[1]+sig*np.array(y),apogee_mu[2]+sig*np.array(z),alpha=.2)

        ax.plot(apogee_mu[0],apogee_mu[1],apogee_mu[2],label="Mean apogee point")

        ax.scatter(0,0,0,marker="x",color="red",label="Launch site")
        ax.set_xlabel("South/m")
        ax.set_ylabel("East/m")
        ax.set_zlabel("Altitude/m")

        if landing_mu.shape!=(0,) and landing_cov.shape!=(0,):
            t=np.linspace(0,2*np.pi,314)

            l_eig=np.linalg.eig(landing_cov)
            l_eig_mat=np.array([[l_eig[1][0][0],l_eig[1][1][0]],[l_eig[1][0][1],l_eig[1][1][1]]])
            landing_cov_elipse = np.matmul(l_eig_mat.transpose(),np.array([l_eig[0][0]**0.5*np.cos(t),l_eig[0][1]**0.5*np.sin(t)]))
            
            for sig in range(1,sigma+1):
                ax.plot(sig*landing_cov_elipse[0]+landing_mu[0],sig*landing_cov_elipse[1]+landing_mu[1],0,label="%s $\sigma$"%sig,linewidth=1)

            ax.scatter(landing_mu[0],landing_mu[1],0,marker="o",s=20,color="black",label="Mean landing point")
            
    ax.set_xlabel('South/m')
    ax.set_ylabel('East/m')
    ax.set_zlabel('Altitude/m') 
    set_axes_equal_3d(ax)
    ax.legend()

    plt.show()

def inertial_position(simulation_output):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x=[]
    y=[]
    z=[]
    for row in simulation_output["pos_i"]:
        x.append(row[0])
        y.append(row[1])
        z.append(row[2])
    ax.plot(x,y,z)
    set_axes_equal_3d(ax)
    plt.show()
