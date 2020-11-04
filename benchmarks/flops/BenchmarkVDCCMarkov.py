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

# ALSO CHANGE
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

# Simulate Markov process for stochastically opening and closing channels
def vdcc_markov_ineff(n_channels, trange, dt):
    states = np.zeros((n_channels, len(trange)), dtype=int)
    n0 = np.zeros(len(trange), dtype=int)
    n1 = np.zeros(len(trange), dtype=int)
    n2 = np.zeros(len(trange), dtype=int)
    n3 = np.zeros(len(trange), dtype=int)
    n4 = np.zeros(len(trange), dtype=int)

    for t_index in range(0, len(trange) - 1):
        for chan_i in range(n_channels):

            # random number between 0 and 1
            r = np.random.rand(1)

            if states[chan_i, t_index] == 0:
                p_trans = [1 - alpha_1(trange[t_index]) * dt, alpha_1(trange[t_index]) * dt, 0, 0, 0]

            elif states[chan_i, t_index] == 1:
                p_trans = [beta_1(trange[t_index]) * dt,
                           1 - (beta_1(trange[t_index]) * dt + alpha_2(trange[t_index]) * dt),
                           alpha_2(trange[t_index]) * dt, 0, 0]

            elif states[chan_i, t_index] == 2:
                p_trans = [0, beta_2(trange[t_index]) * dt,
                           1 - (beta_2(trange[t_index]) * dt + alpha_3(trange[t_index]) * dt),
                           alpha_3(trange[t_index]) * dt, 0]

            elif states[chan_i, t_index] == 3:
                p_trans = [0, 0, beta_3(trange[t_index]) * dt,
                           1 - (beta_3(trange[t_index]) * dt + alpha_4(trange[t_index]) * dt),
                           alpha_4(trange[t_index]) * dt]

            elif states[chan_i, t_index] == 4:
                p_trans = [0, 0, 0, beta_4(trange[t_index]) * dt, 1 - beta_4(trange[t_index]) * dt]

            # state transition
            if r <= np.cumsum(p_trans)[0]:
                states[chan_i, t_index + 1] = 0

            elif r > np.cumsum(p_trans)[0] and r <= np.cumsum(p_trans)[1]:
                states[chan_i, t_index + 1] = 1

            elif r > np.cumsum(p_trans)[1] and r <= np.cumsum(p_trans)[2]:
                states[chan_i, t_index + 1] = 2

            elif r > np.cumsum(p_trans)[2] and r <= np.cumsum(p_trans)[3]:
                states[chan_i, t_index + 1] = 3

            elif r > np.cumsum(p_trans)[3]:
                states[chan_i, t_index + 1] = 4

        num = np.count_nonzero(states[:, t_index] == 0)
        n0[t_index] = num

        num = np.count_nonzero(states[:, t_index] == 1)
        n1[t_index] = num

        num = np.count_nonzero(states[:, t_index] == 2)
        n2[t_index] = num

        num = np.count_nonzero(states[:, t_index] == 3)
        n3[t_index] = num

        num = np.count_nonzero(states[:, t_index] == 4)
        n4[t_index] = num

    return states, n0, n1, n2, n3, n4

# Timing
dt = 1e-3
t_start = 0
t_stop = 10
n_vdcc = 65
trange = np.arange(t_start, t_stop+dt, dt)

# FLOPS
high.start_counters([events.PAPI_DP_OPS,])
vdcc_markov_ineff(n_vdcc, trange, dt)
flops = high.stop_counters()
print(flops)

#high.start_counters([events.PAPI_SP_OPS,])
#vdcc_markov_ineff(n_vdcc, trange, dt)
#flops = high.stop_counters()
#print(flops)
# CHANGE AS NEEDED
#start_time = time.time()
#sim_time = time.time() - start_time

# CHANGE
#with open('bench_{}.txt'.format(seed), 'w') as file:
#    file.write('{0}\t{1}'.format(seed, sim_time))



