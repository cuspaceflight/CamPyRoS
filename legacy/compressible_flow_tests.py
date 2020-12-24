import matplotlib.pyplot as plt
import numpy as np
import trajectory.post as post

#Check post.prandtl_meyer()
print("nu = {} deg for M = 2.8, it should be ~45.75 deg".format(180/np.pi * post.prandtl_meyer(M=2.8)))

#Check post.nu2mach()
print("M = {} for nu = 33.27 deg, it should be ~2.260".format(post.nu2mach(nu = 33.27*np.pi/180)))

#Check if the two Prandtl-Meyer functions agree, by plotting a graph
M = np.linspace(1, 6, 50)
nu = np.linspace(0, 1.5, 50)

nu_derived = []
M_derived = []

for i in range(len(M)):
    nu_derived.append(post.prandtl_meyer(M[i]))
    M_derived.append(post.nu2mach(nu[i]))

plt.plot(M, nu_derived, label = "Exact solution", linewidth = 3)
plt.plot(M_derived, nu, label = "Polynomial approximation for inverse", linewidth = 2)

plt.xlabel("Mach number")
plt.ylabel("Prandtl-Meyer Function (rad)")
plt.legend()
plt.grid()
plt.show()

#Check the compressible flow functions:
M = np.linspace(0, 3, 50)
P = 1e5
P0_derived = []

P0  = 1e5
P_derived = []
for i in range(len(M)):
    P0_derived.append(post.p2p0(P, M[i]))
    P_derived.append(post.p02p(P0, M[i]))

PoverP01 = P/np.array(P0_derived)
PoverP02 = np.array(P_derived)/P0

plt.plot(M, PoverP01, label = "p2p0", linewidth = 3)
plt.plot(M, PoverP02, label = "p02p", linewidth = 2)

plt.xlabel("Mach number")
plt.ylabel("p/p0")
plt.legend()
plt.grid()
plt.show()





