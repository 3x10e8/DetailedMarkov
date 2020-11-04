#!/usr/bin/python3

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time
import sys
from pypapi import events, papi_high as high

def stimulus(fname):
    '''
    Interpolates and creates function for stimulus input from file
    :param fname: stimulus input
    :return: voltage function
    '''

    v_m_df = pd.read_csv(fname, delim_whitespace=True, header=None)
    v_m = v_m_df.to_numpy()

    # interpolate for higher granularity
    vm_cubic_interp = interp1d(v_m[:, 0] * 1000, v_m[:, 1], fill_value="extrapolate")
    v_m = vm_cubic_interp

    return v_m

fname = 'pre_ap_voltage.txt'
v_m = stimulus(fname)

# transition rate functions
# from Bischofberger and Jonas (2002) and Nadkarni et al (2010/2012)
# All in units of ms
# Temperature adjustment from paper
q10 = 2.0  # rxn rate increase with 10 degC temp change; assumuption of uniform
            # q10 for all reactions here
delta_temp = 10.0  # parameters are given at 24C so raise temp by 10C to 34C
                    # (rat internal temp)
temp_comp = q10 ** (delta_temp / 10.)  # change in rxn rate due to deltaT eqtn


# 1ST TRANSITION
v1 = 49.14  # mV


def alpha_1(t):
    '''
    alpha 1 forward transition rate
    :param t: time
    :return: equation for alpha_1
    '''

    a1o = 4.04  # msec-1

    return temp_comp * a1o * np.exp(v_m(t) / v1)


def beta_1(t):
    '''
    beta 1 backward transition rate
    :param t: time int/double value or array
    :return: equation for beta_1
    '''

    b1o = 2.88  # msec-1

    return temp_comp * b1o * np.exp(-v_m(t) / v1)


# 2ND TRANSITION
v2 = 42.08  # mV


def alpha_2(t):
    '''
    alpha 2 forward transition rate
    :param t: time
    :return: equation for alpha_2
    '''

    a2o = 6.70  # msec-1

    return temp_comp * a2o * np.exp(v_m(t) / v2)


def beta_2(t):
    '''
    beta 2 backward transition rate
    :param t: time
    :return: equation for beta_2
    '''

    b2o = 6.30  # msec-1

    return temp_comp * b2o * np.exp(-v_m(t) / v2)


# 3RD TRANSITION
v3 = 55.31  # mV


def alpha_3(t):
    '''
    alpha 3 forward transition rate
    :param t: time
    :return: equation for alpha_3
    '''

    a3o = 4.39  # msec-1

    return temp_comp * a3o * np.exp(v_m(t) / v3)


def beta_3(t):
    '''
    beta 3 backward transition rate
    :param t: time
    :return: equation for beta_3
    '''

    b3o = 8.16  # msec-1

    return temp_comp * b3o * np.exp(-v_m(t) / v3)

# 4TH TRANSITION
v4 = 26.55  # mV


def alpha_4(t):
    '''
    alpha 4 forward transition rate
    :param t: time
    :return: equation for alpha_4
    '''

    a4o = 17.33  # msec-1

    return temp_comp * a4o * np.exp(v_m(t) / v4)


def beta_4(t):
    '''
    beta 4 backward transition rate
    :param t: time
    :return: equation for beta_4
    '''

    b4o = 1.84  # msec-1

    return b4o * np.exp(-v_m(t) / v4)

# CALCIUM INFLUX
# Current influx rate constant (k_ca)
# 1/ms
def k_ca(t):
    f_ca = 511080.18/1802251    # correction factor for only calcium in current
    g_hva = f_ca * 1.55 * 2.4e-12  # conductance (3.72 pS; adjusted for temperature)
    c = 80.36   # mV
    d = 0.3933  # parameter determining current rectification and reversal potential
    e_c = 1.6e-19  # Coulombic charge of one electron (e = F/NA)
    Z_ca = 2  # Charge of calcium ion

    return g_hva * v_m(t) * (d - np.exp(-v_m(t) / c)) / ((1000 ** 2) * Z_ca * e_c * (1 - np.exp(v_m(t) / c)))

seed = int(sys.argv[1])
np.random.seed(seed)
def markov_vdcc_n(n_channels, trange, dt):
    '''
    Simulates a Markov process for the stochastic opening and closing of channels using
    multinomial sampling.

    @param n_channels (int) - number of channels to model
    @param trange (int/float array) - array over which the simulation takes place
    @param dt (int/float) - time step

    @return n_per_state (int array) - number of channels in each state at each time point (shape: [trange, n_states])
    @return ca (int array) - number of calcium that enters at each time point (shape: [trange])
    @return ca_sum (int array) - total sum of calcium that has entered at that time (shape: [trange])
    '''
    n_states = 5
    n_per_state = np.zeros((len(trange), n_states), dtype=int)
    ca = np.zeros(len(trange), dtype=int)
    ca_sum = np.zeros(len(trange), dtype=int)

    # All channels start in C0
    n_per_state[0, 0] = n_channels

    for t_index in range(len(trange) - 1):
        p_trans_0 = [1 - alpha_1(trange[t_index]) * dt, alpha_1(trange[t_index]) * dt, 0, 0, 0]
        samp0 = np.random.multinomial(n_per_state[t_index, 0], p_trans_0)

        p_trans_1 = [beta_1(trange[t_index]) * dt, 1 - (beta_1(trange[t_index]) * dt + alpha_2(trange[t_index]) * dt),
                     alpha_2(trange[t_index]) * dt, 0, 0]
        samp1 = np.random.multinomial(n_per_state[t_index, 1], p_trans_1)

        p_trans_2 = [0, beta_2(trange[t_index]) * dt,
                     1 - (beta_2(trange[t_index]) * dt + alpha_3(trange[t_index]) * dt),
                     alpha_3(trange[t_index]) * dt, 0]
        samp2 = np.random.multinomial(n_per_state[t_index, 2], p_trans_2)

        p_trans_3 = [0, 0, beta_3(trange[t_index]) * dt,
                     1 - (beta_3(trange[t_index]) * dt + alpha_4(trange[t_index]) * dt),
                     alpha_4(trange[t_index]) * dt]
        samp3 = np.random.multinomial(n_per_state[t_index, 3], p_trans_3)

        p_trans_4 = [0, 0, 0, beta_4(trange[t_index]) * dt, 1 - beta_4(trange[t_index]) * dt]
        samp4 = np.random.multinomial(n_per_state[t_index, 4], p_trans_4)

        ca[t_index + 1] = np.random.binomial(n_per_state[t_index, 4], k_ca(trange[t_index]) * dt)
        ca_sum[t_index + 1] = ca[t_index + 1] + ca_sum[t_index]

        n_per_state[t_index + 1, :] = samp0 + samp1 + samp2 + samp3 + samp4

    return n_per_state, ca, ca_sum

# Timing
dt = 1e-3
t_start = 0
t_stop = 10
trange = np.arange(0, t_stop+dt, 1e-3)

"""
start_time = time.time()
n_per_state, ca, ca_sum = markov_vdcc_n(65, trange, dt)
sim_time = time.time() - start_time
print("--- %s seconds ---" % (sim_time))

with open('bench_{}.txt'.format(seed), 'w') as file:
    file.write('{0}\t{1}'.format(seed, sim_time))
"""
# multiple dones
#runtimes = []
#for i in range(int(sys.argv[1])):
#    start_time = time.time()
#    markov_vdcc_n(65, trange, dt)
#    sim_time = time.time() - start_time
#    print("--- %s seconds ---" % (sim_time))
#    runtimes.append(sim_time)

#with open('opt_markov_runtimes.txt', 'w') as file:
#    for rt in runtimes:
#        file.write('{}\n'.format(rt))

# FLOPS
high.start_counters([events.PAPI_DP_OPS,])
markov_vdcc_n(65, trange, dt)
flops = high.stop_counters()
print(flops)

