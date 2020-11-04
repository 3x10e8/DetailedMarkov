#!/usr/bin/python

# import packages
import numpy as np
import sys
import time

'''Constants'''
AVAGADRO = 6.022e23  # avagadro's number
# Axon dimensions
AXON_R = 0.25  # micron
AXON_L = 4  # micron
MICRON_TO_L = 1e-15  # unit conversion
AXON_VOL = np.pi * (AXON_R ** 2) * AXON_L * MICRON_TO_L  # liters
# Calcium
N_CA_AP = 5275  # action potential
CONC_CA_BASE = 1e-7  # ss calcium concentration
N_CA_BASE = int(round(AXON_VOL * CONC_CA_BASE * AVAGADRO))
N_CA = N_CA_AP + N_CA_BASE
CONC_CA_AP = N_CA_AP / AXON_VOL / AVAGADRO
# Calbindin
CONC_CALB = 45e-6  # M
N_CALB = int(round(AXON_VOL * CONC_CALB * AVAGADRO))

#seed = 1
seed = int(sys.argv[1])
np.random.seed(seed)


def calb_markov_ineff(n_calb, n_ca, trange, dt):
    '''Constants'''
    idx_t0 = 0  # initial conditions index
    n_states = 9  # number of calbindin states

    '''Transition rates'''
    # From postsynaptic model paper
    two_bind_sites = 2  # factor for two open binding sites
    k_med_forw = 8.7e7  # medium affinity forward binding (1/Ms)
    k_med_rev = 35.8  # medium affinity reverse binding (1/s)
    k_high_forw = 1.1e7  # high affinity forward binding (1/Ms)
    k_high_rev = 2.6  # high affinity reverse binding (1/s)

    # Convert 1/M s to #/s
    k_med_forw = k_med_forw / (AVAGADRO * AXON_VOL)
    k_high_forw = k_high_forw / (AVAGADRO * AXON_VOL)

    # k_m0m1, k_m1m2, k_m1m0, k_m2m1, k_h0h1, k_h1h2, k_h1h0, k_h2h1
    #   0       1       2       3       4       5       6       7
    k = [two_bind_sites * k_med_forw,  # k[0] = k_M0M1
         k_med_forw,  # k[1] = k_M1M2
         k_med_rev,  # k[2] = k_M1M0
         two_bind_sites * k_med_rev,  # k[3] = k_M2M1
         two_bind_sites * k_high_forw,  # k[4] = k_H0H1
         k_high_forw,  # k[5] = k_H1H2
         k_high_rev,  # k[6] = k_H1H0
         two_bind_sites * k_high_rev]  # k[7] = k_H2H1

    # binding and unbinding for each state
    # h0m0,  h0m1,  h0m2,  h1m0,  h1m1,  h1m2,  h2m0,  h2m1,  h2m2
    #   0     1      2      3      4      5      6      7      8
    k_bind = np.array([k[0] + k[4], k[1] + k[4], k[4],
                       k[0] + k[5], k[1] + k[5], k[5],
                       k[0], k[1], 0])

    k_unbind = np.array([0, k[2], k[3],
                         k[6], k[2] + k[6], k[3] + k[6],
                         k[7], k[2] + k[7], k[3] + k[7]])

    '''Effect of transitions on calcium'''
    delta_ca = np.zeros((n_states, n_states))
    n_state_cols = 3
    ca_loss = -1
    ca_gain = 1
    off_center_shift = 1
    no_remain = 0

    for row in range(n_states):
        for col in range(n_states):
            if (row + n_state_cols) < n_states:
                delta_ca[row, row + n_state_cols] = ca_loss
                delta_ca[row + n_state_cols, row] = ca_gain
            if (row + off_center_shift) < n_states and (row + off_center_shift) % n_state_cols != no_remain:
                delta_ca[row, row + off_center_shift] = ca_loss
                delta_ca[row + off_center_shift, row] = ca_gain

    '''Set initial conditions for all states'''
    # Initializing states
    n_per_state = np.zeros((len(trange), n_states), dtype=int)
    states = np.zeros((n_calb, len(trange)), dtype=int)
    ca = np.zeros(len(trange), dtype=int)

    # Initial amount of calbindin
    # Channels start in ss
    # h0m0,  h0m1,  h0m2,  h1m0,  h1m1,  h1m2,  h2m0,  h2m1,  h2m2
    #   0     1      2      3      4      5      6      7      8
    calb_frac = np.array([0.31958713, 0.15533006, 0.0188739,
                          0.27041988, 0.13143312, 0.01597023,
                          0.0572042, 0.02780316, 0.00337832])  # initial fraction of each calb

    n_per_state[idx_t0, :] = n_calb * calb_frac
    # initialize states
    init_states = []
    for i in range(len(calb_frac)):
        arr = [i] * int(round((calb_frac * n_calb)[i]))
        init_states.extend(arr)

    if len(init_states) > n_calb:
        init_states = init_states[:n_calb]
    states[:, 0] = np.array(init_states)

    # Initial amount of calcium
    ca[idx_t0] = n_ca

    '''Simulation'''
    step = 1
    # All time points except last state
    for t_i in range(len(trange) - 1):
        for c_i in range(n_calb):

            # random number
            r = np.random.rand(1)

            # transition probabilities
            p = np.array(
                [[1 - (k[0] * ca[t_i] * dt + k[4] * ca[t_i] * dt), k[0] * ca[t_i] * dt, 0, k[4] * ca[t_i] * dt, 0,
                  0, 0, 0, 0],

                 [k[2] * dt, 1 - (k[2] * dt + k[1] * ca[t_i] * dt + k[4] * ca[t_i] * dt), k[1] * ca[t_i] * dt, 0,
                  k[4] * ca[t_i] * dt, 0, 0, 0, 0],

                 [0, k[3] * dt, 1 - (k[3] * dt + k[4] * ca[t_i] * dt), 0, 0, k[4] * ca[t_i] * dt, 0, 0, 0],

                 [k[6] * dt, 0, 0, 1 - (k[6] * dt + k[0] * ca[t_i] * dt + k[5] * ca[t_i] * dt),
                  k[0] * ca[t_i] * dt, 0, k[5] * ca[t_i] * dt, 0, 0],

                 [0, k[6] * dt, 0, k[2] * dt,
                  1 - (k[6] * dt + k[2] * dt + k[1] * ca[t_i] * dt + k[5] * ca[t_i] * dt), k[1] * ca[t_i] * dt, 0,
                  k[5] * ca[t_i] * dt, 0],

                 [0, 0, k[6] * dt, 0, k[3] * dt, 1 - (k[6] * dt + k[3] * dt + k[5] * ca[t_i] * dt), 0, 0,
                  k[5] * ca[t_i] * dt],

                 [0, 0, 0, k[7] * dt, 0, 0, 1 - (k[7] * dt + k[0] * ca[t_i] * dt), k[0] * ca[t_i] * dt, 0],

                 [0, 0, 0, 0, k[7] * dt, 0, k[2] * dt, 1 - (k[7] * dt + k[2] * dt + k[1] * ca[t_i] * dt),
                  k[1] * ca[t_i] * dt],

                 [0, 0, 0, 0, 0, k[7] * dt, 0, k[3] * dt, 1 - (k[7] * dt + k[3] * dt)]])

            # state transitions
            p_trans = p[states[c_i, t_i]]

            if r <= np.cumsum(p_trans)[0] and p_trans[0] != 0:
                states[c_i, t_i + 1] = 0

            elif r > np.cumsum(p_trans)[0] and r <= np.cumsum(p_trans)[1] and p_trans[1] != 0:
                states[c_i, t_i + 1] = 1

            elif r > np.cumsum(p_trans)[1] and r <= np.cumsum(p_trans)[2] and p_trans[2] != 0:
                states[c_i, t_i + 1] = 2

            elif r > np.cumsum(p_trans)[2] and r <= np.cumsum(p_trans)[3] and p_trans[3] != 0:
                states[c_i, t_i + 1] = 3

            elif r > np.cumsum(p_trans)[3] and r <= np.cumsum(p_trans)[4] and p_trans[4] != 0:
                states[c_i, t_i + 1] = 4

            elif r > np.cumsum(p_trans)[4] and r <= np.cumsum(p_trans)[5] and p_trans[5] != 0:
                states[c_i, t_i + 1] = 5

            elif r > np.cumsum(p_trans)[5] and r <= np.cumsum(p_trans)[6] and p_trans[6] != 0:
                states[c_i, t_i + 1] = 6

            elif r > np.cumsum(p_trans)[6] and r <= np.cumsum(p_trans)[7] and p_trans[7] != 0:
                states[c_i, t_i + 1] = 7

            elif r > np.cumsum(p_trans)[8] and p_trans[8] != 0:
                states[c_i, t_i + 1] = 8

        for i in range(n_states):
            n_per_state[t_i, i] = len(np.nonzero(states[:, t_i] == i))

        # Multiple sample by calcium change matrix and sum all calcium changes to get overall change
        sample = np.zeros((n_states, n_states))
        ca[t_i + 1] = ca[t_i] + np.sum(np.multiply(sample, delta_ca))

    return states, n_per_state, ca

dt = 1e-6
t_stop = 0.01
t_start = 0
t_range = np.arange(t_start, t_stop + dt, dt)

start_time = time.time() 
calb_markov_ineff(N_CALB, N_CA, t_range, dt)
sim_time = time.time() - start_time

print('{0}\t{1}'.format(seed, sim_time))


