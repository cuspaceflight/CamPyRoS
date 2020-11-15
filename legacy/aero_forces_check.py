import numpy as np
import matplotlib.pyplot as plt
import csv
import main
from main import StandardAtmosphere
from mpl_toolkits.mplot3d import Axes3D

#Import drag coefficients from RasAero II
aerodynamic_coefficients = main.RasAeroData("Martlet4 RasAeroII.CSV")

#Import motor data - copied from Joe Hunt's simulation
with open('motor_out.csv') as csvfile:
    motor_out = csv.reader(csvfile)

    (motor_time_data, prop_mass_data, cham_pres_data,
     throat_data, gamma_data, nozzle_efficiency_data,
     exit_pres_data, area_ratio_data) = [], [], [], [], [], [], [], []

    next(motor_out)
    for row in motor_out:
        motor_time_data.append(float(row[0]))
        prop_mass_data.append(float(row[1]))
        cham_pres_data.append(float(row[2]))
        throat_data.append(float(row[3]))
        gamma_data.append(float(row[4]))
        nozzle_efficiency_data.append(float(row[5]))
        exit_pres_data.append(float(row[6]))
        area_ratio_data.append(float(row[7]))      

#Create the HybridMotor object
pulsar = main.Motor(motor_time_data, prop_mass_data, cham_pres_data, throat_data,
                          gamma_data, nozzle_efficiency_data, exit_pres_data, area_ratio_data)

#Create the LaunchSite
launch_site = main.LaunchSite(10, 10, 0, [0,0,0], StandardAtmosphere)

#Create the Rocket
martlet4 = main.Rocket(45.73, 86.8, 86.8, 0.32, pulsar, aerodynamic_coefficients, launch_site)

#Try inputting some velocity values and check if the resulting aero forces make sense
v_b = np.array([100, -40,20])
martlet4.v_b = v_b

#Plot the vectors in 3D, so we can see visually if things make sense
fig3d = plt.figure(figsize=(15,15))
ax3d = fig3d.add_subplot(111, projection='3d')

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      
    Source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to 
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

#Rocket is shown as a transparent cylinder
height = 6.38
radius = 0.0985
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

Xc,Yc,Zc = data_for_cylinder_along_z(0,0,radius,height)
ax3d.plot_surface(Xc, Yc, Zc, alpha=0.3)

#Get the aero forces
f, cop = martlet4.aero_forces() 
print("Forces = {} N, \n COP = {} m = {} inches".format(f, cop, cop*39.3701))

#Plot the COP as a point
ax3d.scatter(0,0,height - cop, s=500, color='black', label="COP")

#Velocity is in red - note we shifted x,y,z to become y,z,x so that the rocket points in the z-direciton in our plot
ax3d.quiver(0,0,height - cop,
          v_b[1]/20,v_b[2]/20,v_b[0]/20,
          color = 'red',alpha =1, lw = 3, label="Velocity")

#Plot x,y,z forces
ax3d.quiver(0,0,height-cop,
          0,0,f[0]/30,
          color='black')

ax3d.quiver(0,0,height-cop,
          f[1]/30,0,0,
          color='black')

ax3d.quiver(0,0,height-cop,
          0,f[2]/30,0,
          color='black')

#Net aerodynamic force is shown in green
ax3d.quiver(0,0,height-cop,
          f[1]/30,f[2]/30,f[0]/30,
          color='green', label="Net aerodynamic force")

ax3d.legend()
set_axes_equal(ax3d)
plt.show()


#Now vary the velocity with alpha = 0, and plot graphs of the resulting forces 
v_bx = np.linspace(0,5*300, 500)
fx = np.zeros(len(v_bx))
fy = np.zeros(len(v_bx))
fz = np.zeros(len(v_bx))

for i in range(len(v_bx)):
    martlet4.v_b = np.array([v_bx[i], 0, 0])
    f, cop = martlet4.aero_forces()
    fx[i], fy[i], fz[i] = f[0], f[1], f[2]


fig2d = plt.figure(figsize=(15,15))
ax2d = fig2d.add_subplot(111)
ax2d.plot(v_bx, fx, label="fx")
ax2d.plot(v_bx, fy, label="fy")    
ax2d.plot(v_bx, fz, label="fz")   
ax2d.legend()
ax2d.grid()
ax2d.set_xlabel("velocity / ms^-1")
ax2d.set_ylabel("force / N")
ax2d.set_title("Angle of attack = 0")
plt.show()

#Now fix the velocity at 600m/s, and plot a graph of forces against angle of attack
alpha = np.linspace(-0.10472, 0.10472, 100) #Go from alpha = -6 to 6 degrees (Numpy uses radians usually)
alpha_deg = 360/(2*np.pi) * alpha
v = 600
fx = np.zeros(len(alpha))
fy = np.zeros(len(alpha))
fz = np.zeros(len(alpha))

for i in range(len(alpha)):
    v_bx = v*np.cos(alpha[i])
    v_by = v*np.sin(alpha[i])
    
    martlet4.v_b = np.array([v_bx, v_by, 0])
    f, cop = martlet4.aero_forces()
    fx[i], fy[i], fz[i] = f[0], f[1], f[2]

fig2d_alpha = plt.figure(figsize=(15,15))
ax2d_alpha = fig2d_alpha.add_subplot(111)
ax2d_alpha.plot(alpha_deg, fx, label="fx")
ax2d_alpha.plot(alpha_deg, fy, label="fy")    
ax2d_alpha.plot(alpha_deg, fz, label="fz")   
ax2d_alpha.legend()
ax2d_alpha.grid()
ax2d_alpha.set_xlabel("angle of attack / degrees")
ax2d_alpha.set_ylabel("force / N")
ax2d_alpha.set_title("Velocity = 600 ms^-1")
plt.show()
