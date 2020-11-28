"""6DOF Martlet trajectory simulator"""
'''Contains classes and functions used to run trajectory simulations'''
'''All units in SI unless otherwise stated'''

'''
COORDINATE SYSTEM NOMENCLATURE
x_b,y_b,z_b = Body coordinate system (origin on rocket, rotates with the rocket)
x_i,y_i,z_i = Inertial coordinate system (does not rotate, origin at centre of the Earth)
x_l, y_l, z_l = Launch site coordinate system - the origin is on launch site but dropped down to a position of zero altitude (whilst keeping the same long and lat). Rotates with the Earth.

Directions are defined below.
- Body:
    y points east and z north at take off (before rail alignment is accounted for) x up.
    x is along the "long" axis of the rocket.

- Launch site:
    z points perpendicular to the earth, y in the east direction and x tangential to the earth pointing south
    
- Inertial:
    Origin at centre of the Earth
    z points to north from centre of earth, x aligned with launchsite at start and y orthogonal

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def get_velocity_magnitude(df):
    return (np.sqrt(df["vx_l"]**2+df["vy_l"]**2+df["vz_l"]**2))

def plot_altitude_time(simulation_output):

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(simulation_output["y_l"], -simulation_output["x_l"])
    axs[0, 0].set_title('Ground Track ($^*$)')
    axs[0,0].set_xlabel("East/m")
    axs[0,0].set_ylabel("North/m")
    #plt.text(0,-simulation_output["x"].max(),'$^*$ This is in the fixed cartesian launch frame so will not be actual ground position over large distances',horizontalalignment='left', verticalalignment='center')
    axs[0, 1].plot(simulation_output["time"],simulation_output["z_l"], 'tab:orange')
    axs[0, 1].set_title('Altitude')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Altitude/m")
    axs[1, 0].plot(simulation_output["time"],simulation_output.apply(get_velocity_magnitude,axis=1), 'tab:green')
    axs[1, 0].set_title('Speed')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Speed/m/s")
    axs[1, 1].plot(simulation_output["time"],simulation_output["vz_l"], 'tab:red')
    axs[1, 1].set_title('Vertical Velocity')
    axs[1,1].set_xlabel("time/s")
    axs[1,1].set_ylabel("Velocity/m/s")
    fig.tight_layout()

    plt.show() 

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

def plot_ypr(simulation_output):
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(simulation_output["time"], simulation_output["yaw"])
    axs[0, 0].set_title('Yaw')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Angles/ rad")
    
    axs[0, 1].plot(simulation_output["time"], simulation_output["pitch"])
    axs[0, 1].set_title('Pitch')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Angles/ rad")
    
    axs[1, 0].plot(simulation_output["time"], simulation_output["roll"])
    axs[1, 0].set_title('Roll')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Angles/ rad")

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
    
def plot_launch_trajectory_3d(simulation_output, show_orientation=False, show_aero=False, arrow_frequency = 0.02):
    '''
    Plots the trajectory in 3D, given the simulation_output
    '''
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    x=simulation_output["x_l"]
    y=simulation_output["y_l"]
    z=simulation_output["z_l"]
    
    
    #Plot rocket position and launch site position
    ax.plot3D(x,y,z)
    ax.scatter(x[0],y[0],z[0],c='red', label="Launch site", linewidths="10")
    ax.set_xlabel('South')
    ax.set_ylabel('East')
    ax.set_zlabel('Altitude')  

    
    #Indenxes to plot arrows at
    idx = np.round(np.linspace(0, len(x) - 1, arrow_frequency*int(len(x)))).astype(int)
    
    #Plot the direction the rocket faces at each point (i.e. direction of x_b), using quivers
    if show_orientation==True:
        u=simulation_output["attitude_xl"]
        v=simulation_output["attitude_yl"]
        w=simulation_output["attitude_zl"]

        ax.quiver(x[idx], y[idx], z[idx], u[idx], v[idx], w[idx], length=1000, normalize=True, color="red", label="Orientation")
        
    if show_aero ==True:
        aero_x = simulation_output["aero_xl"]
        aero_y = simulation_output["aero_yl"]
        aero_z = simulation_output["aero_zl"]
        
        ax.quiver(x[idx], y[idx], z[idx], aero_x[idx], 0, 0, length=1000, normalize=True, color="black", label = "Aerodynamic forces")
        ax.quiver(x[idx], y[idx], z[idx], 0, aero_y[idx], 0, length=1000, normalize=True, color="black")
        ax.quiver(x[idx], y[idx], z[idx], 0, 0, aero_z[idx], length=1000, normalize=True, color="black")
 
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
    
    ax.legend()
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
    burnout_time = simulation_output["burnout_time"]
    

    
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
    burnout_index = (np.abs(time - burnout_time)).idxmin()
    axs[1,1].scatter(time[burnout_index], altitude[burnout_index], color="red", label="Engine burnout")
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
    
    axs[1,1].set_xlim(0,200)
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
