"""Nitrous oxide vapour pressure fed hybrid rocket motor firing simulator"""

### Joe Hunt updated 20/06/19 ###
### All units SI unless otherwise stated ###

import csv
import numpy as np
import matplotlib.pyplot as plt
import hybrid_functions as motor

###############################################################################
# Input parameters
###############################################################################

VOL_TANK = 60*0.001           #tank volume (m^3)
HEAD_SPACE = 0.1              #initial vapour phase proportion

# primary injectory orifices were drilled by shaped machining
NUM_INJ1 = 40                 #number of primary injector orifices
DIA_INJ1 = 0.0013             #diameter of primary injector orifices (m)

# secondary injector orifices were drilled prior to first test
NUM_INJ2 = 4                  #number of secondary injector orifices
DIA_INJ2 = 0.002              #diameter of secondary injector orifices (m)

# tertiary injectory orifices are new orifices to be drilled
NUM_INJ3 = 8                  # number of tertiary injector orifices
DIA_INJ3 = 0.0015             #diameter of tertiary injector orifices (m)

DIA_PORT = 0.075              #diameter of fuel port (m)
LENGTH_PORT = 1.33            #length of fuel port (m)
DIA_FUEL = 0.112              #Outside diameter of fuel grain (m)
C_STAR_EFFICIENCY = 0.95      # Ratio between actual and theoretical characteristic velocity

DIA_THROAT = 0.0432           # nozzle throat diameter (m)
NOZZLE_EFFICIENCY = 0.97      # factor by which to reduce thrust coefficient
NOZZLE_AREA_RATIO = 4.5       # ratio of nozzle exit area to throat area

DIA_FEED = 0.02               #feed pipe diameter (m)
LENGTH_FEED = 0.5             #feed pipe length (m)
VALVE_MODEL_TYPE = 'ball'     # either 'kv' or 'ball' (models as thick orifice)
KV_VALVE = 5                  # used if VALVE_MODEL_TRY='kv'
DIA_VALVE = 0.015             # used if VALVE_MODEL_TRY='ball'
LENGTH_VALVE = 0.08           # used if VALVE_MODEL_TRY='ball'

DENSITY_FUEL = 935            #solid fuel density (kg m^-3)
REG_COEFF = 1.157E-4	      #regression rate coefficient (usually 'a' in textbooks)
REG_EXP = 0.331		      #regression rate exponent (usually 'n' in textbooks)

PRES_EXTERNAL = 101325        #external atmospheric pressure at test site (Pa)
temp = 20+273.15              #initial tank temperature (K)


###############################################################################
# Initialize simulation
###############################################################################

STEP = 0.01 # time step (s)

#open propep data file
propep_file = open('L_Nitrous_S_HDPE.propep', 'r')
propep_data = propep_file.readlines()

#open compressibility_data csv file
with open('n2o_compressibility_factors.csv') as csvfile:
    compressibility_data = csv.reader(csvfile)
    pdat, zdat = motor.compressibility_read(compressibility_data)

# assign initial values
vapz_lag = 0; time = 0; mdotox = 0; impulse = 0; gamma_N2O = 1.31; blowdown_type = 'liquid'
lden, vden, hl, hg, cp, vap_pres, ldynvis = motor.thermophys(temp) #temperature dependent properties
hv = hg-hl # spec heat of vapourization
pres_cham = PRES_EXTERNAL

#calculate initial propellant masses
lmass = VOL_TANK*(1-HEAD_SPACE)*lden
vmass = VOL_TANK*HEAD_SPACE*vden
fuel_mass = ((((DIA_FUEL/2)**2)*np.pi-(((DIA_PORT/2)**2)*np.pi))
             * LENGTH_PORT*DENSITY_FUEL)
tmass = lmass+vmass

# create empty lists to fill with output data
(time_data, vap_pres_data, pres_cham_data, thrust_data, gox_data,
 prop_mass_data, manifold_pres_data, gamma_data, throat_data,
 nozzle_efficiency_data, exit_pressure_data,
 area_ratio_data) = ([], [], [], [], [], [], [], [], [], [], [], [])

#additional properties needed for the 6DOF simulation
(vden_data, vmass_data, lden_data, lmass_data, fuel_mass_data) = ([], [], [], [], [])

# print initial conditions
print("Initial conditions:\ntime:", time, "s\ntank temperature:", temp-273.15,
      "C\nlmass:", lmass, "kg\nvmass:", vmass, "kg\nvap_pres:", vap_pres,
      'Pa\nfuel thickness:', (DIA_FUEL-DIA_PORT)/2, 'm\nfuel mass',
      fuel_mass, 'kg\n')

###############################################################################
# Simulation loop
###############################################################################

while True:

    time += STEP  #increment time


    # calculate feed system losses (only attemped for liquid phase)
    if mdotox > 0 and lmass > 0:
        flow_speed = mdotox/(lden*(((DIA_FEED/2)**2)*np.pi))
        entry_loss = (1/2)*lden*(flow_speed**2) # loss at tank entry

        reynolds = (lden*flow_speed*DIA_FEED)/ldynvis
        f = motor.Nikuradse(reynolds)
        vis_pdrop = (f*lden*(flow_speed**2)*LENGTH_FEED)/(4*DIA_FEED) # loss in pipe

        if VALVE_MODEL_TYPE == 'ball':
            #valve loss from full bore ball valve modelled as thick orifice
            valve_loss = ((1/2)*lden*(flow_speed**2)
                          *motor.ball_valve_K(reynolds, DIA_FEED, DIA_VALVE, LENGTH_VALVE))
        if VALVE_MODEL_TYPE == 'kv':
            valve_loss = 1.296E9*(mdotox**2)/(lden*(KV_VALVE**2))

        manifold_pres = vap_pres-entry_loss-valve_loss-vis_pdrop # sum pressure drops
    else:
        manifold_pres = vap_pres


    #calculate injector pressure drop
    inj_pdrop = manifold_pres-pres_cham

    if inj_pdrop < 0.15 and time > 0.5:
        print('FAILURE: Reverse flow occurred at t=', time, 's')
        break

    # model tank emptying

    if blowdown_type == 'liquid':
        #liquid phase blowdown

        mdotox1 = NUM_INJ1 * motor.dyer_injector(pres_cham, DIA_INJ1, lden, inj_pdrop,
                                                 hl, manifold_pres, vap_pres)
        mdotox2 = NUM_INJ2 * motor.dyer_injector(pres_cham, DIA_INJ2, lden, inj_pdrop,
                                                 hl, manifold_pres, vap_pres)
        mdotox3 = NUM_INJ3 * motor.dyer_injector(pres_cham, DIA_INJ3, lden, inj_pdrop,
                                                 hl, manifold_pres, vap_pres)
        mdotox = mdotox1 + mdotox2 + mdotox3 # sum flow from 3 types of orifice

        tmass = tmass-(mdotox*STEP) # find new mass of tank contents after outflow

        lmass_pre_vap = lmass-(mdotox*STEP) # liquid mass prior to vaporization
        lmass_post_vap = ((VOL_TANK-(tmass/vden))/((1/lden)-(1/vden))) #lmass post vaporization

        if lmass_pre_vap < lmass_post_vap: # check for liquid depletion
            print('starting vapour blowdown, vapour mass is', vmass+lmass, 'kg')
            print('injector pressure drop at liquid depletion was',
                  (inj_pdrop/pres_cham)*100, '%')
            blowdown_type = 'vapour'
            lmass = 0
            vmass = tmass
            #define tank parameters at liquid depletion
            vmass_ld, temp_ld, vden_ld, vap_pres_ld = vmass, temp, vden, vap_pres
            Z_ld = np.interp(motor.thermophys(temp_ld)[5], pdat, zdat)

        else: # continue with liquid blowdown stage
            lmass = lmass_post_vap
            vapz = lmass_pre_vap-lmass #mass vapourized
            #add 1st order lag of 0.15s to model vaporization time
            vapz_lag = (STEP/0.15)*(vapz-vapz_lag)+vapz_lag
            vmass = tmass-lmass

            #update nitrous thermophysical properties given new temperature
            temp = temp-((vapz_lag*hv)/(lmass*cp))
            lden, vden, hl, hg, cp, vap_pres, ldynvis = motor.thermophys(temp)
            hv = hg-hl # spec heat of vapourization

    else:
        # vapour phase blowdown

        #calculations for injector orifices
        mdotox1 = NUM_INJ1 * motor.vapour_injector(DIA_INJ1, vden, inj_pdrop)
        mdotox2 = NUM_INJ2 * motor.vapour_injector(DIA_INJ2, vden, inj_pdrop)
        mdotox3 = NUM_INJ3 * motor.vapour_injector(DIA_INJ3, vden, inj_pdrop)
        mdotox = mdotox1+mdotox2+mdotox3
        vmass -= STEP*mdotox # sum flow from 3 types of orifice

        #find current tank vapour parameters
        Z2 = motor.Z2_solve(temp_ld, Z_ld, vmass_ld, vmass, gamma_N2O, zdat, pdat)
        if Z2 == 'numerical instability':
            print('vapour depleted: finishing motor simulation')
            break
        temp = temp_ld*(((Z2*vmass)/(Z_ld*vmass_ld))**(gamma_N2O-1)) #isentropic assumption
        vap_pres = vap_pres_ld*((temp/temp_ld)**(gamma_N2O/(gamma_N2O-1)))
        vden = vden_ld*((temp/temp_ld)**(1/(gamma_N2O-1)))

    # check for excessive mass flux
    if mdotox/(((DIA_PORT/2)**2)*np.pi) > 600:
        print('Failure: oxidizer flux too high:',
              mdotox/(((DIA_PORT/2)**2)*np.pi))
        break

    # fuel port calculation
    rdot = (REG_COEFF)*((mdotox/(((DIA_PORT/2)**2)*np.pi))**REG_EXP)
    mdotfuel = rdot*DENSITY_FUEL*(np.pi*DIA_PORT*LENGTH_PORT)
    DIA_PORT += 2*rdot*STEP
    if DIA_PORT > DIA_FUEL: #check for depleted fuel grain
        print("fuel depleted")
        break
    fuel_mass = ((((DIA_FUEL/2)**2)*np.pi)-(((DIA_PORT/2)**2)*np.pi))*LENGTH_PORT*DENSITY_FUEL


    # lookup characteristic velocity using previous pres_cham and current OF from propep data
    c_star = motor.c_star_lookup(pres_cham, mdotox/mdotfuel, propep_data)
    c_star = c_star * C_STAR_EFFICIENCY

    # calculate current chamber pressure
    pres_cham = ((mdotox+mdotfuel)*c_star)/(((DIA_THROAT/2)**2)*np.pi)

    # lookup ratio of specific heats from propep data file
    gamma = motor.gamma_lookup(pres_cham, mdotox/mdotfuel, propep_data)

    # performance calculations
    # find nozzle exit static pressure
    mach_exit = motor.mach_exit(gamma, NOZZLE_AREA_RATIO)
    pres_exit = pres_cham*(1+(gamma-1)*mach_exit**2/2)**(-gamma/(gamma-1))


    # motor performance calculations
    area_throat = ((DIA_THROAT/2)**2)*np.pi
    thrust = (area_throat*pres_cham*(((2*gamma**2/(gamma-1))
                                     *((2/(gamma+1))**((gamma+1)/(gamma-1)))
                                     *(1-(pres_exit/pres_cham)**((gamma-1)/gamma)))**0.5)
             +(pres_exit-PRES_EXTERNAL)*area_throat*NOZZLE_AREA_RATIO)

    thrust *= NOZZLE_EFFICIENCY


    #update data lists
    time_data.append(time)
    vap_pres_data.append(vap_pres)
    pres_cham_data.append(pres_cham)
    manifold_pres_data.append(manifold_pres)
    thrust_data.append(thrust)
    gox_data.append(mdotox/(((DIA_PORT/2)**2)*np.pi))
    prop_mass_data.append(lmass+vmass+fuel_mass)
    gamma_data.append(gamma)
    throat_data.append(DIA_THROAT)
    nozzle_efficiency_data.append(NOZZLE_EFFICIENCY)
    exit_pressure_data.append(pres_exit)
    area_ratio_data.append(NOZZLE_AREA_RATIO)

    #additional data for the 6DOF simulation
    vmass_data.append(vmass)
    vden_data.append(vden)
    lden_data.append(lden)
    lmass_data.append(lmass)
    fuel_mass_data.append(fuel_mass)
    


###############################################################################
# Print and plot results
###############################################################################

#print final results
print("\nFinal conditions:\ntime:", time, "s\ntank temperature:", temp-273.15,
      "C\nlmass:", lmass, "kg\nvmass:", vmass, "kg\nvap_pres:", vap_pres,
      'Pa\nfuel thickness:', (DIA_FUEL-DIA_PORT)/2, 'm\nfuel mass', fuel_mass,
      'kg')

impulse = 0
for i in range(len(time_data)):
    impulse += STEP*thrust_data[i]

print('\nPerformance results:\nInitial thrust:', thrust_data[int(0.5/STEP)],
      'N\nmean thrust:', np.mean(thrust_data), 'N\nimpulse:', impulse,
      'Ns\nmean Isp:', impulse/(prop_mass_data[0]-fuel_mass)/9.81)

#plot pressures
plt.figure(figsize=(8.5, 7))
plt.subplot(221)
plt.plot(time_data, vap_pres_data, 'b', label='Tank pressure')
plt.plot(time_data, pres_cham_data, 'r', label='Chamber pressure')
plt.plot(time_data, manifold_pres_data, 'g', label='Injector manifold pressure')
plt.ylabel('Pressure (Pa)')
plt.ylim(0, max(vap_pres_data)*1.3)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.legend()
plt.tight_layout()

#plot thrust
plt.subplot(222)
plt.plot(time_data, thrust_data)
plt.xlabel('Time (s)')
plt.ylabel('thrust (N)')
plt.ylim(0, max(thrust_data)*1.3)
plt.tight_layout()

#plot massflux
plt.subplot(223)
plt.plot(time_data, gox_data, 'y')
plt.xlabel('Time (s)')
plt.ylabel('Oxidizer mass flux ($kg s^{-1} m^{-2}$)')
plt.ylim(0, max(gox_data)*1.3)
plt.tight_layout()

#plot mass of propellant
plt.subplot(224)
plt.plot(time_data, prop_mass_data, 'g')
plt.xlabel('Time (s)')
plt.ylabel('Propellant mass (kg)')
plt.ylim(0, max(prop_mass_data)*1.3)
plt.tight_layout()

plt.show()

###############################################################################
# generate motor_output.csv for trajectory simulation
###############################################################################
with  open("motor_out.csv", "w", newline='') as motor_file:
    motor_file.truncate()
    motor_write = csv.writer(motor_file)
    motor_write.writerow(['Time', 'Propellant mass (kg)',
                          'Chamber pressure (Pa)', 'Throat diameter (m)',
                          'Nozzle inlet gamma', 'Nozzle efficiency',
                          'Exit static pressure (Pa)', 'Area ratio', 
                          'Vapour Density (kg/m^3)', 'Vapour Mass (kg)', 'Liquid Density (kg/m^3)', 'Liquid Mass (kg)', 'Solid Fuel Mass (kg)',
                          'Solid Fuel Density (kg/m^3)', 'Solid Fuel Outer Diameter (m)', 'Solid Fuel Length (m)'])

    for i in range(len(time_data)):
        motor_write.writerow([time_data[i], prop_mass_data[i], pres_cham_data[i],
                              throat_data[i], gamma_data[i],
                              nozzle_efficiency_data[i], exit_pressure_data[i],
                              area_ratio_data[i], 
                              vden_data[i], vmass_data[i], lden_data[i], lmass_data[i], fuel_mass_data[i],
                              DENSITY_FUEL, DIA_FUEL, LENGTH_PORT])

    motor_write.writerow([time_data[-1]+STEP, fuel_mass, pres_cham_data[-1],
                          throat_data[-1], gamma_data[-1],
                          nozzle_efficiency_data[-1], exit_pressure_data[-1],
                          area_ratio_data[-1],
                          vden_data[-1], 0, lden_data[-1], lmass_data[-1], fuel_mass_data[-1],
                          DENSITY_FUEL, DIA_FUEL, LENGTH_PORT])

###############################################################################
# generate a RASP motor file for RAS Aero
###############################################################################

RASP_DIA = 160                    # motor diameter in mm
RASP_LENGTH = 3000                # motor length in mm
RASP_DRY = 40                     # motor dry mass in kg

rasp_file = open("hybrid.eng", "w+")

rasp_file.write(';\n')
header_line = ''
header_line += 'Pulsar'+' '
header_line += str(RASP_DIA)+' '
header_line += str(RASP_LENGTH)+' '
header_line += 'P'+' '
header_line += str(round(prop_mass_data[0], 2))+' '
header_line += str(round((prop_mass_data[0]+RASP_DRY), 2))+' '
header_line += 'CUSF'+'\n'
rasp_file.write(header_line)

for i in range(31):
    performance_line = '   '
    performance_line += str(round(time_data[int(i*len(time_data)/31)], 2))+' '
    performance_line += str(round(thrust_data[int(i*len(time_data)/31)], 2))+'\n'
    rasp_file.write(performance_line)

performance_line = '   '+str(round((time_data[-1]), 2))+' 0.0 \n'
rasp_file.write(performance_line)
rasp_file.write(';')

rasp_file.close()
