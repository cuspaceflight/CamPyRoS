import trajectory,csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

run_name="stat_model_20201203"
itterations = 1

fig, axs = plt.subplots(2, 2)
axs[0,1].scatter(0,0,marker="x",s=5)

xyz = []

for n in range(1,itterations+1):
    run_data = pd.read_csv("results/%s/%s.csv"%(run_name,n))

    xyz.append([run_data["x"], run_data["y"], run_data["z"]])
    speed = [(v_x**2+run_data["v_y"][index]**2+run_data["v_z"][index]**2)**.5 for index, v_x in enumerate(run_data["v_x"])]

    axs[1,0].plot(run_data["time"],speed,linewidth=1)
    axs[0,0].plot(run_data["time"],run_data["z"],linewidth=1)
    axs[0,1].scatter(run_data["x"].values[-1],run_data["y"].values[-1],marker="x",s=5)
    axs[1,1].plot(run_data["time"],run_data["v_z"],linewidth=1)
    

#Make all the axes scales equal
#trajectory.set_axes_equal(axs[0,0])
fig.tight_layout()
axs[0,0].set_title("Altitude")
axs[0,0].set_xlabel("Time/s")
axs[0,0].set_ylabel("Altitude/m")
axs[1,0].set_title("Speed")
axs[1,0].set_xlabel("Time/s")
axs[1,0].set_ylabel("Speed/m/s")
axs[0,1].set_title("Downrange")
axs[0,1].set_xlabel("North/m")
axs[0,1].set_ylabel("East/m")
axs[1,1].set_title("Vertical Velocity")
axs[1,1].set_xlabel("Time/s")
axs[1,1].set_ylabel("Velocity/m/s")
plt.show() 

fig = plt.figure()
ax = plt.axes(projection='3d')
for itt in xyz:
    ax.plot3D(itt[0], itt[1], itt[2],linewidth=1)
ax.set_xlabel('South/m')
ax.set_ylabel('East/m')
ax.set_zlabel('Altitude/m')  
ax.legend()

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


