import trajectory
import numpy as np
import matplotlib.pyplot as plt

REF_AREA = 0.0305128422             #Reference area for aerodynamic coefficients (m^2)
C_DAMP_PITCH = 0.0
C_DAMP_ROLL = 0.0

#Import drag coefficients from RASAero II
aero_data = trajectory.AeroData.from_rasaero("data/Martlet4RASAeroII.CSV", REF_AREA, C_DAMP_PITCH, C_DAMP_ROLL)


Mach = np.linspace(0, 25, 500)
alpha = np.linspace(0, 4, 5)
alpha = alpha*np.pi/180

CA = []
CN = []
COP = []

for j in range(len(alpha)):
    #For each angle of attack
    CA.append([])
    CN.append([])
    COP.append([])

    for i in range(len(Mach)):
        #For each Mach number
        CA[j].append(aero_data.CA(Mach[i], alpha[j]))
        CN[j].append(aero_data.CN(Mach[i], alpha[j]))
        COP[j].append(aero_data.COP(Mach[i], alpha[j]))
        print(f"Mach = {Mach[i]}, alpha = {alpha[j]} rad")

alpha = alpha*180/np.pi

for i in range(len(alpha)):
    plt.plot(Mach, COP[i], label=f"alpha = {alpha[i]} deg")

plt.grid()
plt.legend()
plt.show()

