#!/usr/bin/python

import numpy as np
from scipy.optimize import fsolve
import time

'''Calbindin concentrations as constant number'''
AXON_RAD = 0.25    # micron
AXON_LEN = 4       # micron
CONC_CALB = 45e-6  # M
CONC_CA_BASE = 1e-7 # M
MICRON_TO_L = 1e-15   # unit conversion
AVAGADRO = 6.022e23   # avagadro's number
N_CA_AP = 5275     # avg ca after ap

# find axon volume
axon_vol_um3 = np.pi*(AXON_RAD**2)*AXON_LEN    # micron^3
axon_vol = axon_vol_um3*MICRON_TO_L    # liters

# convert concentrations to constant numbers
n_calb = int(round(axon_vol * CONC_CALB * AVAGADRO))
n_ca_base = int(round(axon_vol * CONC_CA_BASE * AVAGADRO))

# convert number of calcium after ap to concentration
conc_ca_ap = N_CA_AP/(axon_vol * AVAGADRO)
conc_ca = CONC_CA_BASE + conc_ca_ap # M

def calb_ss(calb):
    '''
     ODEs to describe calbindin kinetics according to Bartol et al (2015)
     input:        calb:        array of calbindin concentrations (arr, 9 x 1)
     output:       eqtns:          the blender user-set value of the parameter (float or None)
     '''

    # k_values meaning
    TWO_BIND_SITES = 2  # factor for two open binding sites
    K_MED_FOR = 8.7e7  # medium affinity forward binding
    K_MED_REV = 35.8  # medium affinity reverse binding
    K_HIGH_FOR = 1.1e7  # high affinity forward binding
    K_HIGH_REV = 2.6  # high affinity reverse binding

    # k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h2, k_h1h0, k_h2h1
    CALB_K_VALS = [TWO_BIND_SITES * K_MED_FOR,
                   K_MED_FOR,
                   K_MED_REV,
                   TWO_BIND_SITES * K_MED_REV,
                   TWO_BIND_SITES * K_HIGH_FOR,
                   K_HIGH_FOR,
                   K_HIGH_REV,
                   TWO_BIND_SITES * K_HIGH_REV]

    # Unpack input
    h0m0, h0m1, h0m2, h1m0, h1m1, h1m2, h2m0, h2m1, h2m2 = calb
    k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h0, k_h2h1 = CALB_K_VALS

    # calcium
    ca = CONC_CA_BASE

    eqtns = []

    eqtns.append(-(k_h0h1 * ca + k_m0m1 * ca) * h0m0 + k_m1m0 * h0m1 + k_h1h0 * h1m0)

    eqtns.append(-(k_m1m0 + k_h0h1 * ca + k_m1m2 * ca) * h0m1 + k_m0m1 * ca * h0m0 + k_m2m1 * h0m2 + k_h1h0 * h1m1)

    eqtns.append(-(k_h0h1 * ca + k_m2m1) * h0m2 + k_h1h0 * h1m2 + k_m1m2 * ca * h0m1)

    eqtns.append(-(k_h1h0 + k_m0m1 * ca + k_h1h2 * ca) * h1m0 + k_m1m0 * h1m1 + k_h2h1 * h2m0 + k_h0h1 * ca * h0m0)

    eqtns.append(-(k_m1m0 + k_h1h0 + k_m1m2 * ca + k_h1h2 * ca) * h1m1 + k_h2h1 * h2m1 + k_m2m1 * h1m2
                 + k_m0m1 * ca * h1m0 + k_h0h1 * ca * h0m1)

    eqtns.append(-(k_m2m1 + k_h1h0 + k_h1h2 * ca) * h1m2 + k_h2h1 * h2m2 + k_m1m2 * ca * h1m1 + k_h0h1 * ca * h0m2)

    eqtns.append(-(k_h2h1 + k_m0m1 * ca) * h2m0 + k_m1m0 * h2m1 + k_h1h2 * ca * h1m0)

    eqtns.append(-(k_h2h1 + k_m1m0 + k_m1m2 * ca) * h2m1 + k_m2m1 * h2m2 + k_m0m1 * ca * h2m0 + k_h1h2 * ca * h1m1)

    eqtns.append(-(k_h2h1 + k_m2m1) * h2m2 + k_m1m2 * ca * h2m1 + k_h1h2 * ca * h1m2)

    return eqtns

'''Solve for steady-state solutions of ODE'''
N_STATES = 9
TWO_BIND_SITES = 2 # factor for two open binding sites
K_MED_FOR = 8.7e7    # medium affinity forward binding
K_MED_REV = 35.8     # medium affinity reverse binding
K_HIGH_FOR = 1.1e7    # high affinity forward binding
K_HIGH_REV = 2.6      # high affinity reverse binding

# k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h2, k_h1h0, k_h2h1
CALB_K_VALS = [TWO_BIND_SITES*K_MED_FOR, K_MED_FOR, K_MED_REV,
               TWO_BIND_SITES*K_MED_REV, TWO_BIND_SITES*K_HIGH_FOR,
               K_HIGH_FOR, K_HIGH_REV, TWO_BIND_SITES*K_HIGH_REV]


conc_calb_x0 = np.ones(N_STATES)    # starting estimate for roots

# Solve for roots of system of nonlinear equations
conc_calb = fsolve(calb_ss, conc_calb_x0)

# Fraction of calibnindin in each state at ss
# h0m0, h0m1, h0m2, h1m0, h1m1, h1m2, h2m0, h2m1, h2m2
calb_frac = np.empty(N_STATES)
for i in range(N_STATES):
    calb_frac[i] = conc_calb[i]/sum(conc_calb)

seed = 1
def calb_euler():
    # k values
    # k_values meaning
    TWO_BIND_SITES = 2 # factor for two open binding sites
    K_MED_FOR = 8.7e7    # medium affinity forward binding (M ca/sec)
    K_MED_REV = 35.8     # medium affinity reverse binding
    K_HIGH_FOR = 1.1e7    # high affinity forward binding
    K_HIGH_REV = 2.6      # high affinity reverse binding

    # k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h2, k_h1h0, k_h2h1
    CALB_K_VALS = [TWO_BIND_SITES *K_MED_FOR,
                   K_MED_FOR,
                   K_MED_REV,
                   TWO_BIND_SITES *K_MED_REV,
                   TWO_BIND_SITES *K_HIGH_FOR,
                   K_HIGH_FOR,
                   K_HIGH_REV,
                   TWO_BIND_SITES *K_HIGH_REV]

    k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h0, k_h2h1 = CALB_K_VALS

    # Initial conditions
    y0 = CONC_CALB *calb_frac    # IC in M
    y0 = np.append(y0, conc_ca)

    # Time
    # dt = 0.000025
    # t_stop = 0.005
    dt = 1e-6
    t_stop = 0.01
    t_start = 0
    t = np.arange(t_start, t_stop +dt, dt)

    # Initialize states
    h0m0, h0m1, h0m2, h1m0, h1m1, h1m2, h2m0, h2m1, h2m2, ca = [np.zeros(len(t)) for _ in range(len(y0))]

    # Initial conditions
    h0m0[0] = y0[0]
    h0m1[0] = y0[1]
    h0m2[0] = y0[2]
    h1m0[0] = y0[3]
    h1m1[0] = y0[4]
    h1m2[0] = y0[5]
    h2m0[0] = y0[6]
    h2m1[0] = y0[7]
    h2m2[0] = y0[8]
    ca[0] = y0[9]


    # Step forward in time
    for n in range(len(t) - 1):
        h0m0[n+1] = h0m0[n] + (-(k_h0h1 * ca[n] + k_m0m1 * ca[n]) * h0m0[n] +
                                k_m1m0 * h0m1[n] + k_h1h0 * h1m0[n]) * dt

        h0m1[n+1] = h0m1[n] + (-(k_m1m0 + k_h0h1 * ca[n] + k_m1m2 * ca[n]) *
                                 h0m1[n] + k_m0m1 * ca[n] * h0m0[n] + k_m2m1 *
                                 h0m2[n] + k_h1h0 * h1m1[n]) * dt

        h0m2[n+1] = h0m2[n] + (-(k_h0h1 * ca[n] + k_m2m1) * h0m2[n] + k_h1h0
                                 * h1m2[n] + k_m1m2 * ca[n] * h0m1[n]) * dt

        h1m0[n+1] = h1m0[n] + (-(k_h1h0 + k_m0m1 * ca[n] + k_h1h2 * ca[n]) *
                                 h1m0[n] + k_m1m0 * h1m1[n] + k_h2h1 * h2m0[n]
                                 + k_h0h1 * ca[n] * h0m0[n]) * dt

        h1m1[n+1] = h1m1[n] + (-(k_m1m0 + k_h1h0 + k_m1m2 * ca[n] + k_h1h2 *
                                 ca[n]) * h1m1[n] + k_h2h1 * h2m1[n] + k_m2m1 *
                               h1m2[n] + k_m0m1 * ca[n] * h1m0[n] + k_h0h1 *
                               ca[n] * h0m1[n]) * dt

        h1m2[n+1] = h1m2[n] + (-(k_m2m1 + k_h1h0 + k_h1h2 * ca[n]) * h1m2[n] +
                               k_h2h1 * h2m2[n] + k_m1m2 * ca[n] * h1m1[n] +
                               k_h0h1 * ca[n] * h0m2[n]) * dt

        h2m0[n+1] = h2m0[n] + (-(k_h2h1 + k_m0m1 * ca[n]) * h2m0[n] +
                               k_m1m0 * h2m1[n] + k_h1h2 * ca[n] * h1m0[n]) * dt

        h2m1[n+1] = h2m1[n] + (-(k_h2h1 + k_m1m0 + k_m1m2 * ca[n]) * h2m1[n] +
                               k_m2m1 * h2m2[n] + k_m0m1 * ca[n] * h2m0[n] +
                               k_h1h2 * ca[n] * h1m1[n]) * dt

        h2m2[n+1] = h2m2[n] + (-(k_h2h1 + k_m2m1) * h2m2[n] + k_m1m2 * ca[n] *
                               h2m1[n] + k_h1h2 * ca[n] * h1m2[n]) * dt

        ca[n+1] = ca[n] + (-(k_h0h1 * (h0m0[n] + h0m1[n] + h0m2[n]) + k_h1h2 *
                             (h1m0[n] + h1m1[n] + h1m2[n]) + k_m0m1 *
                             (h0m0[n] + h1m0[n] + h2m0[n]) + k_m1m2 *
                             (h0m1[n] + h1m1[n] + h2m1[n])) * ca[n] + k_h1h0 *
                           (h1m0[n] + h1m1[n] + h1m2[n]) + k_h2h1 *
                           (h2m0[n] + h2m1[n] + h2m2[n]) + k_m1m0 *
                           (h0m1[n] + h1m1[n] + h2m1[n]) + k_m2m1 *
                           (h0m2[n] + h1m2[n] + h2m2[n])) * dt

    # Normalize
    h0m0 = h0m0 / CONC_CALB
    h0m1 = h0m1 / CONC_CALB
    h0m2 = h0m2 / CONC_CALB
    h1m0 = h1m0 / CONC_CALB
    h1m1 = h1m1 / CONC_CALB
    h1m2 = h1m2 / CONC_CALB
    h2m0 = h2m0 / CONC_CALB
    h2m1 = h2m1 / CONC_CALB
    h2m2 = h2m2 / CONC_CALB
    ca = ca / conc_ca

start_time = time.time()
calb_euler()
sim_time = time.time() - start_time

print('{0}\t{1}'.format(seed, sim_time))
