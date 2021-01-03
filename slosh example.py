import trajectory, trajectory.slosh, trajectory.post, thermo.chemical, scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

N2O = thermo.chemical.Chemical('N2O')
print("Fuel = {}".format(N2O))

#Fill the fuel tanks to various heights
tank_length = 2.5
fuel_95 = trajectory.slosh.CylindricalFuelTank(197e-3, 0.95*tank_length, N2O.rhol)
fuel_75 = trajectory.slosh.CylindricalFuelTank(197e-3, 0.75*tank_length, N2O.rhol)
fuel_50 = trajectory.slosh.CylindricalFuelTank(197e-3, 0.5*tank_length, N2O.rhol)
fuel_25 = trajectory.slosh.CylindricalFuelTank(197e-3, 0.25*tank_length, N2O.rhol)

print("\nFor a 50% full fuel tank:")
print('ω_spring = {:.4f} rad/s'.format(fuel_50.w_spring()))
print('ω_pendulum = {:.4f} rad/s'.format(fuel_50.w_pendulum()))

#Now let's try find out what sort of frequencies we're dealing with during flight
imported_data = trajectory.from_json("trajectory.json")

#Lets find the yaw, pitch and roll as a function of time
yaw, pitch, roll = trajectory.post.ypr_i(imported_data)
time = np.array(imported_data["time"])

#FFT can only be applied to data with uniform timestamps. We need to convert our data into one with uniform timestamps.
#Only go up to around 50 seconds, as this is approximately when apogee is reached.
#We'll use the yaw data and analyse that, since it has a nice wobble to it.
y_function = scipy.interpolate.interp1d(time, yaw, 'quadratic')

#Now do the FFT - modified from https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html#d-discrete-fourier-transforms
#Number of sample points
N = 6000
#Sample spacing
T = 1.0 / 150

#Data
x = np.linspace(0.1, N*T, N, endpoint=False)
y = y_function(x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

#Plot everything
fig, axs = plt.subplots(1, 2)

axs[0].plot(x, y, label="yaw")
axs[0].grid()
axs[0].legend()
axs[0].set_xlabel("Time (s)")

axs[1].plot(xf, 2.0/N * np.abs(yf[0:N//2]), label="Fourier Transform", color="orange")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].axvline(x=fuel_25.w_spring()/(2*np.pi), label="ω_spring (25% full)", color = "cornflowerblue", linestyle="--")
axs[1].axvline(x=fuel_50.w_spring()/(2*np.pi), label="ω_spring (50% full)", color = "royalblue", linestyle="--")
axs[1].axvline(x=fuel_75.w_spring()/(2*np.pi), label="ω_spring (75% full)", color = "blue", linestyle="--")
axs[1].axvline(x=fuel_95.w_spring()/(2*np.pi), label="ω_spring (95% full)", color = "darkblue", linestyle="--")

axs[1].legend()
axs[1].grid()

plt.show()

