"""6DOF Trajectory Simulator

Various useful plots of the outputted data

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l

def get_velocity_magnitude(df):
    return (np.sqrt(df["vx_l"]**2+df["vy_l"]**2+df["vz_l"]**2))

#Functional
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to 

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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
    
    #Plot rocket position and launch site position
    ax.plot3D(x_l, y_l, z_l)
    ax.scatter(x_l[0], y_l[0], z_l[0], c='red', label="Launch site", linewidths="10")
    ax.set_xlabel('South')
    ax.set_ylabel('East')
    ax.set_zlabel('Altitude')  
    
    #Indexes to plot arrows at
    idx = np.round(np.linspace(0, len(x_l) - 1, arrow_frequency*int(len(x_l)))).astype(int)
    
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
    set_axes_equal(ax)
    
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

#Non-functional

def plot_aero_forces(simulation_output):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(simulation_output["time"], simulation_output["aero_xb"])
    axs[0, 0].set_title('aero_x_b')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Force/N")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["aero_yb"])
    axs[0, 1].set_title('aero_y_b')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Force/N")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["aero_zb"])
    axs[1, 0].set_title('aero_z_b')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Force/N")
    
    axs[1, 1].plot(simulation_output["time"], -simulation_output["cop"], label="CoP")
    axs[1, 1].plot(simulation_output["time"], -simulation_output["cog"], label="CoG")
    axs[1, 1].set_title('Centre of Pressure (positive relative to nose tip)')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Distance/m")
    
    axs[1,1].legend()
    
    plt.show()
    
def plot_velocity(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["vx_l"])
    axs[0, 0].set_title('V_x')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Velocity/m/s")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["vy_l"])
    axs[0, 1].set_title('V_y')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Velocity/m/s")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["vz_l"])
    axs[1, 0].set_title('V_z')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Velocity/m/s")
    plt.show()

def plot_position(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["x_l"])
    axs[0, 0].set_title('x')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Distance/m")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["y_l"])
    axs[0, 1].set_title('y')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Distance/m")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["z_l"])
    axs[1, 0].set_title('z')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Distance/m")
    plt.show()

def fix_ypr(point):
    """if point<0:
        point = 2*np.pi-abs(point)
    point = round(point,5)
    if point==round(2*np.pi,5):
        point=0"""
    return point

def plot_ypr(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], [fix_ypr(n) for n in simulation_output["yaw"]])
    axs[0, 0].set_title('Yaw')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Angles/ rad")
    
    axs[0, 1].plot(simulation_output["time"], [fix_ypr(n) for n in simulation_output["pitch"]])
    axs[0, 1].set_title('Pitch')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Angles/ rad")
    
    axs[1, 0].plot(simulation_output["time"], [fix_ypr(n) for n in simulation_output["roll"]])
    axs[1, 0].set_title('Roll')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Angles/ rad")

    axs[1, 1].plot(simulation_output["time"], simulation_output["z_l"])
    axs[1, 1].set_title('Altitude')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Altitude /m")
    
    plt.show()

def plot_attitude(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["attitude_xl"])
    axs[0, 0].set_title('X')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Distance/m")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["attitude_yl"])
    axs[0, 1].set_title('Y')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Distance/m")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["attitude_zl"])
    axs[1, 0].set_title('Z')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Distance/m")

    axs[1, 1].plot(simulation_output["time"], simulation_output["z_l"])
    axs[1, 1].set_title('Altitude')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Altitude /m")
    
    plt.show()

def plot_w_b(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["w_bx"])
    axs[0, 0].set_title('w_bx')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Angular velocity/ rad s^-1")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["w_by"])
    axs[0, 1].set_title('w_by')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Angular velocity/ rad s^-1")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["w_bz"])
    axs[1, 0].set_title('w_bz')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Angular velocity/ rad s^-1")
    
    plt.show()

def plot_quat_i2b(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["quat_i2b[0]"])
    axs[0, 0].set_title('quat_i2b[0]')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Value")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["quat_i2b[1]"])
    axs[0, 1].set_title('quat_i2b[1]')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Value")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["quat_i2b[2]"])
    axs[1, 0].set_title('quat_i2b[2]')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Value")

    axs[1, 1].plot(simulation_output["time"], simulation_output["quat_i2b[3]"])
    axs[1, 1].set_title('quat_i2b[3]')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Value")
    
    plt.show()

def plot_quat_i2bdot(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["quat_i2bdot[0]"])
    axs[0, 0].set_title('quat_i2bdot[0]')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Value")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["quat_i2bdot[1]"])
    axs[0, 1].set_title('quat_i2bdot[1]')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Value")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["quat_i2bdot[2]"])
    axs[1, 0].set_title('quat_i2bdot[2]')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Value")

    axs[1, 1].plot(simulation_output["time"], simulation_output["quat_i2bdot[3]"])
    axs[1, 1].set_title('quat_i2bdot[3]')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Value")
    
    plt.show()

def plot_wdot_b(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["wdot_bx"])
    axs[0, 0].set_title('wdot_bx')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Angular acceleration/ rad s^-2")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["wdot_by"])
    axs[0, 1].set_title('wdot_by')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Angular acceleration/ rad s^-2")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["wdot_bz"])
    axs[1, 0].set_title('wdot_bz')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Angular acceleration/ rad s^-2")
    
    plt.show()

def plot_inertial_trajectory_3d(simulation_output, show_orientation=False):
    '''
    Plots the trajectory in 3D, given the simulation_output
    '''
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    x=simulation_output["x_i"]
    y=simulation_output["y_i"]
    z=simulation_output["z_i"]
    
    #Plot rocket position and launch site position
    ax.plot3D(x,y,z)
    ax.scatter(x[0],y[0],z[0],c='red', label="Launch site")
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')  
    ax.legend()
    
    #Plot the direction the rocket faces at each point (i.e. direction of x_b), using quivers
    if show_orientation==True:
        u=simulation_output["attitude_xi"]
        v=simulation_output["attitude_yi"]
        w=simulation_output["attitude_zi"]
        
        #Spaced out arrows, so it's not cluttered
        idx = np.round(np.linspace(0, len(u) - 1, int(len(u)/30))).astype(int)
        
        ax.quiver(x[idx], y[idx], z[idx], u[idx], v[idx], w[idx], length=1000, normalize=True, color="red")
        
    #Make axes equal ratios - source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to 
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    plt.show() 
     
def animate_orientation(simulation_output, frames=500):
    '''frames : number of animation frames in total - less means that the animations runs faster'''

    fig, axs = plt.subplots(2, 2)
    
    #Get data
    yaw=simulation_output["yaw"]
    pitch=simulation_output["pitch"]
    roll=simulation_output["roll"]
    altitude = simulation_output["z_l"]

    time = simulation_output["time"]
    #burnout_time = simulation_output["burnout_time"]
    

    
    #Add titles
    axs[0, 0].set_title('Yaw')
    axs[0, 1].set_title('Pitch')
    axs[1, 0].set_title('Roll')
    axs[1, 1].set_title('Altitude / m')
    axs[1, 1].set_xlabel('time / s')
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()    
    
    #Plot the initial directions
    axs[0,0].plot(np.linspace(0,np.cos(yaw[0]), 100), np.linspace(0, np.sin(yaw[0]), 100), lw=3, color='black')
    axs[0,1].plot(np.linspace(0,np.cos(pitch[0]), 100), np.linspace(0, np.sin(pitch[0]), 100), lw=3, color='black')
    axs[1,0].plot(np.linspace(0,np.cos(roll[0]), 100), np.linspace(0, np.sin(roll[0]), 100), lw=3, color='black')

    #Plot the point of engine burnout on the altitude-time graph
    #burnout_index = (np.abs(time - burnout_time)).idxmin()
    #axs[1,1].scatter(time[burnout_index], altitude[burnout_index], color="red", label="Engine burnout")
    axs[1,1].legend()
    
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
    
    def animate1(i):
        j = int(i*len(yaw)/frames)
        x = np.linspace(0,np.cos(yaw[j]), 100)
        y = np.linspace(0, np.sin(yaw[j]), 100)
        line1.set_data(x, y)
        return line1,
    
    def animate2(i):
        j = int(i*len(pitch)/frames)
        x = np.linspace(0,np.cos(pitch[j]), 100)
        y = np.linspace(0, np.sin(pitch[j]), 100)
        line2.set_data(x, y)
        return line2,
    
    def animate3(i):
        j = int(i*len(roll)/frames)
        x = np.linspace(0,np.cos(roll[j]), 100)
        y = np.linspace(0, np.sin(roll[j]), 100)
        line3.set_data(x, y)
        return line3,
    
    def animate4(i):
        j = int(i*len(altitude)/frames)
        y = altitude[0:j]
        x = time[0:j]
        line4.set_data(x, y)
        return line4,
    
    
    anim1 = FuncAnimation(fig, animate1, init_func=init1,
                                   frames=frames, interval=20, blit=True)
    anim2 = FuncAnimation(fig, animate2, init_func=init2,
                                   frames=frames, interval=20, blit=True)
    anim3 = FuncAnimation(fig, animate3, init_func=init3,
                                   frames=frames, interval=20, blit=True)
    anim4 = FuncAnimation(fig, animate4, init_func=init4,
                                   frames=frames, interval=20, blit=True)
    
    
    plt.show()
