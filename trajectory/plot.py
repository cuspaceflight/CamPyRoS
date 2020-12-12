"""6DOF Trajectory Simulator

Various useful plots of the outputted data

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import trajectory
from trajectory.transforms import pos_l2i, pos_i2l, vel_l2i, vel_i2l, direction_l2i, direction_i2l, pos_i2alt, i2airspeed
from scipy.spatial.transform import Rotation

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

def fix_ypr(point):
    """if point<0:
        point = 2*np.pi-abs(point)
    point = round(point,5)
    if point==round(2*np.pi,5):
        point=0"""
    return point

def plot_ypr(simulation_output, rocket):
    #Get data
    output_dict = simulation_output.to_dict(orient="list")
    
    yaw=[]
    pitch=[]
    roll=[]
    z_l=[]
    for index, row in simulation_output.iterrows():#this is ugly but proper way not working
        ypr=trajectory.Rotation.from_matrix(row["b2imat"]).as_euler("zyx")
    altitude=[]
    time = output_dict["time"]

    for i in range(len(time)):
        ypr = Rotation.from_matrix(output_dict["b2imat"][i]).as_euler("zyx")
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
    burnout_time = rocket.motor.motor_time_data[-1]
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

    aero_x_b = []
    aero_y_b = []
    aero_z_b = []
    q_data = []

    cop_data = []
    cog_data = []

    #Angles of attack (as defined in https://apps.dtic.mil/sti/pdfs/AD0642855.pdf)
    alpha_star_data = []
    beta_data = []
    delta_data = []

    for i in range(len(output_dict["time"])):
        b2i = Rotation.from_matrix(output_dict["b2imat"][i])

        aero_b, cop, q = rocket.aero_forces(output_dict["pos_i"][i], output_dict["vel_i"][i], b2i, output_dict["w_b"][i], output_dict["time"][i])
        aero_x_b.append(aero_b[0])
        aero_y_b.append(aero_b[1])
        aero_z_b.append(aero_b[2])
        q_data.append(q/1000)

        cop_data.append(-cop)
        cog_data.append(-rocket.mass_model.cog(output_dict["time"][i]))

        v_rel_wind = b2i.inv().apply( direction_l2i((i2airspeed(output_dict["pos_i"][i], output_dict["vel_i"][i], rocket.launch_site, output_dict["time"][i]) - rocket.launch_site.wind), rocket.launch_site,  output_dict["time"][i]) )
        beta = np.arctan2(v_rel_wind[1], (v_rel_wind[0]**2 + v_rel_wind[2]**2 )**0.5 )
        delta = np.arctan2((v_rel_wind[2]**2 + v_rel_wind[1]**2)**0.5, v_rel_wind[0])
        alpha_star = np.arctan2(v_rel_wind[2], (v_rel_wind[0]**2 + v_rel_wind[1]**2 )**0.5 )

        alpha_star_data.append(alpha_star*180/np.pi)
        beta_data.append(beta*180/np.pi)
        delta_data.append(delta*180/np.pi)

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
        altitude.append(pos_i2alt(output_dict["pos_i"][i]))

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
    