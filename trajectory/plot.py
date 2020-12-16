"""6DOF Trajectory Simulator

Various useful plots of the outputted data

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import trajectory
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l
import pandas as pd

def get_velocity_magnitude(df):
    return (np.sqrt(df["vx_l"]**2+df["vy_l"]**2+df["vz_l"]**2))

#Functional
def set_axes_equal(ax,dim=3):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to 

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if dim==3:
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
    elif dim==2:
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range])

        ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])

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
    ax.scatter(x_l[0], y_l[0], z_l[0], c='red', label="Launch site", linewidths=10)
    ax.set_xlabel('South')
    ax.set_ylabel('East')
    ax.set_zlabel('Altitude')  
    
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

    return fig

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

def plot_ypr(simulation_output, rocket):
    yaw=[]
    pitch=[]
    roll=[]
    z_l=[]
    for index, row in simulation_output.iterrows():#this is ugly but proper way not working
        ypr=trajectory.Rotation.from_matrix(row["b2imat"]).as_euler("zyx")
        yaw.append(ypr[0])
        pitch.append(ypr[1])
        roll.append(ypr[2])
        z_l.append(trajectory.pos_i2l(np.array(row["pos_i"]),rocket.launch_site,row["time"])[2])
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

def plot_attitude(simulation_output,rocket,launch_site):
    attitude=[]
    for index, row in simulation_output.iterrows():
        x_b_l = trajectory.direction_i2l(trajectory.Rotation.from_matrix(row["b2imat"]).apply([1,0,0]), launch_site, row["time"])
        new_row={"attitude_xl":x_b_l[0],
                "attitude_yl":x_b_l[1],
                "attitude_zl":x_b_l[2],
                "z_l":trajectory.pos_i2l(np.array(row["pos_i"]),rocket.launch_site,row["time"])[2],
                "time":row["time"]}
        attitude.append(new_row)
    attitude=trajectory.pd.DataFrame(attitude) 

    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(attitude["time"], attitude["attitude_xl"])
    axs[0, 0].set_title('X')
    axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("Distance/m")
    
    axs[0, 1].plot(attitude["time"], attitude["attitude_yl"])
    axs[0, 1].set_title('Y')
    axs[0,1].set_xlabel("time/s")
    axs[0,1].set_ylabel("Distance/m")
    
    axs[1, 0].plot(attitude["time"], attitude["attitude_zl"])
    axs[1, 0].set_title('Z')
    axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("Distance/m")

    axs[1, 1].plot(attitude["time"], attitude["z_l"])
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
    set_axes_equal(ax,dim=2)
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
            d=elipse(u,v,ap_eig[0][2]**.5,ap_eig[0][1]**.5,ap_eig[0][0]**.5)
            d=np.matmul(ap_eig_mat,d)
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
        landing_cov_elipse = np.matmul(l_eig_mat,np.array([l_eig[0][0]**0.5*np.cos(t),l_eig[0][1]**0.5*np.sin(t)]))
        
        for sig in range(1,sigma+1):
            ax.plot(sig*landing_cov_elipse[0]+landing_mu[0],sig*landing_cov_elipse[1]+landing_mu[1],0,label="%s $\sigma$"%sig,linewidth=1)

        ax.scatter(landing_mu[0],landing_mu[1],0,marker="o",s=20,color="black",label="Mean landing point")

        if not landing.empty:
            ax.scatter(landing.x,landing.y,0,marker="o",s=10,color="blue",alpha=0.3)


    set_axes_equal(ax)
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
            plt.plot([incriments*n for n in range(0,len(mean_alts))],np.array(mean_alts)-np.array(st_dev)*sig,alpha=(.9-.1*sig),color="orange",label="%s$\sigma$"%sig,linewidth=1)
    plt.xlabel("Time/s")
    plt.ylabel("Altitude/m")
    plt.legend()
    plt.ylim(0,None)
    plt.show()