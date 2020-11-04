'''
Script for looking at VDCC transitions and calcium influx based on equations
from Bischofberger and Jone (2002). Similar to script used for Nadkarni pub.
Temperature is adjusted for physiologically relevant temperature. IV curve
adjusted according to Nadkarni (to account for only calcium influx)

Creates figures comparing different simulation outputs
'''


# import packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
from math import ceil
import glob
from scipy.integrate import solve_ivp
import seaborn as sns



# load action potential time series


def stimulus(fname):
    '''
    Interpolates and creates function for stimulus input from file
    :param fname: stimulus input
    :return: voltage function
    '''

    v_m = np.loadtxt(fname)
    v_m = interp1d(v_m[:, 0] * 1000, v_m[:, 1], kind='cubic', fill_value="extrapolate")

    return v_m


fname = './pre_ap_voltage.txt'
v_m = stimulus(fname)


####################################
###### TRANSITION RATES ############
####################################
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

##################################
######## ODE SIMULATION ##########
##################################


def vdcc_odes(t, state):
    '''
    ODEs to describe vdcc states according to Markov diagram
    :param t: time
    :param state: states of vdcc
    :return: odes describing each state
    '''

    c0, c1, c2, c3, o, ca = state

    eqtns = []
    eqtns.append(beta_1(t) * c1 - alpha_1(t) * c0)
    eqtns.append(alpha_1(t) * c0 + beta_2(t) * c2 - (beta_1(t) + alpha_2(t)) * c1)
    eqtns.append(alpha_2(t) * c1 + beta_3(t) * c3 - (beta_2(t) + alpha_3(t)) * c2)
    eqtns.append(alpha_3(t) * c2 + beta_4(t) * o - (beta_3(t) + alpha_4(t)) * c3)
    eqtns.append(alpha_4(t) * c3 - beta_4(t) * o)
    eqtns.append(k_ca(t) * o)

    return eqtns


dt = 1e-3
t_start = 0
t_stop = 10
trange = np.arange(t_start, t_stop+dt, dt)
ode_results = solve_ivp(vdcc_odes, [t_start, t_stop+dt], [1, 0, 0, 0, 0, 0],
                        t_eval=trange)

##################################
###### EULER SIMULATION ##########
##################################

def euler(y0, t_start, t_stop, dt):
    trange = np.arange(t_start, t_stop + dt, dt)

    # Initialize states
    vdcc_c0, vdcc_c1, vdcc_c2, vdcc_c3, vdcc_o, ca = [np.zeros(len(trange)) for _ in range(len(y0))]

    # Initial conditions
    vdcc_c0[0] = y0[0]
    vdcc_c1[0] = y0[1]
    vdcc_c2[0] = y0[2]
    vdcc_c3[0] = y0[3]
    vdcc_o[0] = y0[4]
    ca[0] = y0[5]

    # Step forward in time
    for n in range(len(trange) - 1):
        # print(n)
        vdcc_c0[n + 1] = vdcc_c0[n] + (beta_1(trange[n]) * vdcc_c1[n] - alpha_1(trange[n]) * vdcc_c0[n]) * dt
        vdcc_c1[n + 1] = vdcc_c1[n] + (alpha_1(trange[n]) * vdcc_c0[n] + beta_2(trange[n]) * vdcc_c2[n] - (
                    beta_1(trange[n]) + alpha_2(trange[n])) * vdcc_c1[n]) * dt
        vdcc_c2[n + 1] = vdcc_c2[n] + (alpha_2(trange[n]) * vdcc_c1[n] + beta_3(trange[n]) * vdcc_c3[n] - (
                    beta_2(trange[n]) + alpha_3(trange[n])) * vdcc_c2[n]) * dt
        vdcc_c3[n + 1] = vdcc_c3[n] + (alpha_3(trange[n]) * vdcc_c2[n] + beta_4(trange[n]) * vdcc_o[n] - (
                    beta_3(trange[n]) + alpha_4(trange[n])) * vdcc_c3[n]) * dt
        vdcc_o[n + 1] = vdcc_o[n] + (alpha_4(trange[n]) * vdcc_c3[n] - beta_4(trange[n]) * vdcc_o[n]) * dt
        ca[n + 1] = ca[n] + (k_ca(trange[n]) * vdcc_o[n]) * dt

    return vdcc_c0, vdcc_c1, vdcc_c2, vdcc_c3, vdcc_o, ca

# Initial conditions
y0 = [1, 0, 0, 0, 0, 0]

# Time
dt = 1e-3
t_start = 0
t_stop = 10


#vdcc_c0, vdcc_c1, vdcc_c2, vdcc_c3, vdcc_o, ca = euler(y0, t_start, t_stop, dt)

##################################
###### MARKOV SIMULATION #########
##################################
# Simulate Markov process for stochastically opening and closing channels

def markov_vdcc_n(n_channels, trange, dt):
    '''
    Simulates a Markov process for the stochastic opening and closing of
     channels using multinomial sampling.
    :param n_channels: number of channels to model (int)
    :param trange: array over which the simulation takes place
    :param dt: time step
    :return: n_per_state - number of channels in each state at each time point
                            (shape: [trange, n_states])
             ca (int array) - number of calcium that enters at each time point
                            (shape: [trange])
             ca_sum (int array) - total sum of calcium that has entered at that time
                            (shape: [trange])
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

##################################
#### RUN MARKOV SIMULATION #######
##################################
"""
def time(seconds):
    '''
    Convert seconds into hours, minutes and seconds
    :param seconds: number of seconds
    :return: hours:minutes:seconds
    '''

    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

# Constants
N_RUNS = 1000
N_VDCC_STATES = 5
N_CHANS = 65

# Time array
dt = 1e-3
trange = np.arange(0, 10+dt, dt)

# Initialize array sizes
ca = np.empty([len(trange), N_RUNS])
ca_sum = np.empty([len(trange), N_RUNS])
vdcc_states = np.empty([len(trange), N_VDCC_STATES, N_RUNS])

# Expected time for simulation to run
print("This simulation should take approximately", time(7*N_RUNS), "to run.")

# RUN SIMULATION
for i in range(N_RUNS):
    vdcc_states[:, :, i], ca[:, i], ca_sum[:, i] = markov_vdcc_n(N_CHANS, 
    trange, dt)

# SAVE OUTPUT
# trial number for each time point
trial = np.empty(N_RUNS * len(trange))

for run in range(N_RUNS):
    for tp in range(len(trange)):
        trial[len(trange) * run + tp] = run + 1

# repeat timepoints for each trial
timepoints = np.tile(trange, N_RUNS)

markov_results = pd.DataFrame(trial, columns=['trial'])
markov_results['timepoints'] = timepoints
for i in range(4):
    results['vdcc_c{:d}'.format(i)] = vdcc_states[:, i, :].flatten('F')
markov_results['vdcc_o'] = vdcc_states[:, 4, :].flatten('F')
markov_results['ca_sum'] = ca_sum.flatten('F')
markov_results['ca'] = ca.flatten('F')

MAX_STORE = 1000000

# Save data
for i in range(int(ceil(markov_results.shape[0]/MAX_STORE))):
    # Save to csv
    markov_results.iloc[i*MAX_STORE:(i+1)*MAX_STORE].to_csv('markov_1000_runs_{:02d}.csv'.format(i), index=False) # CHECK THIS LINE 02d
"""


##########################
###### MARKOV DATA #######
##########################
def markov_sim(fname):
    N_CHANS = 65
    sep_results_df = (pd.read_csv(file) for file in sorted(glob.glob(fname)))
    markov_results = pd.concat(sep_results_df)

    # Normalize
    for col_name in markov_results.columns[2:8]:
        markov_results['{}_norm'.format(col_name)] = markov_results[col_name] / N_CHANS

    # Average across trials
    avg_markov_results = markov_results.groupby('timepoints').mean()
    del avg_markov_results['trial']

    markov_results.head()

    return markov_results, avg_markov_results

fname = "markov_1000_runs*.csv"
markov_results, avg_markov_results = markov_sim(fname)




##################################
####### MCELL SIMULATION #########
##################################
##########################
####### MCELL DATA #######
##########################
def mcell_sim(fname):
    SEEDS = 50
    DATA_DIM = 2
    T_START = 0
    ITERATIONS = 10000
    T_STEP = 1e-6
    T_STOP = ITERATIONS * T_STEP
    T_RANGE = np.arange(T_START, T_STOP + T_STEP / 2, T_STEP)

    # Molecule data file names (ie ca.World.dat)
    mol_files = []
    mol_names = []
    files = sorted(glob.glob(os.path.join(MCELL_DIR, "seed_00001/vdcc_pre_*.World.dat")))
    files[4], files[5] = files[5], files[4]

    for file_path in files:
        mol_file = file_path.split('/')[-1]
        mol_files.append(mol_file)
        mol_names.append(mol_file.split('.')[0])

    # Initialize data arrays
    # Data is list of np arrays n_mol in length
    mcell_data = []
    for i in range(len(mol_files)):
        mcell_data.append(np.empty([len(T_RANGE), DATA_DIM, SEEDS]))
        # time x features (2 time, val) x seeds

    # Add data to all arrays
    for seed in range(SEEDS):
        for i in range(len(mol_files)):
            fname = os.path.join(MCELL_DIR, "seed_{:05d}".format(seed + 1), mol_files[i])
            mcell_data[i][:, :, seed] = pd.read_csv(fname, delim_whitespace=True, header=None)

    mcell_results = np.copy(mcell_data)
    N_CHANS = 65
    for i in range(len(mcell_results)):
        # if i != 5:
        mcell_results[i][:, 1, :] = mcell_results[i][:, 1, :] / N_CHANS

    return mcell_results

MCELL_DIR = "/Users/margotwagner/projects/mcell/simple_geom/model_1/" \
            "model_1_vdcceqtns_tchange_nodendrite_files/mcell/output_data/" \
            "react_data/"

mcell_results = mcell_sim(MCELL_DIR)

##################################
###### COMPARE SIMULATIONS #######
##################################
def vdcc_plot(odesolve=False, euler=False, mcell=False, markov=False, incl_std=False):
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    N_CHANS = 65

    names = markov_results.columns[9:].values

    for a, i in zip(ax.flatten(), range(ode_results.y.shape[0])):
        #########
        # MCELL #
        #########
        if mcell == True:

            mcell_color = 'C4'
            mcell_mean = np.mean(mcell_results[i], axis=2)
            mcell_std = np.std(mcell_results[i], axis=2)

            # Plot mean
            a.plot(mcell_mean[:, 0] * 1000, mcell_mean[:, 1], color=mcell_color, label='MCell')

            # Cloud plot for std dev
            if incl_std == True:
                a.fill_between(mcell_mean[:, 0] * 1000, np.add(mcell_mean[:, 1], mcell_std[:, 1]),
                               np.subtract(mcell_mean[:, 1], mcell_std[:, 1]), alpha=0.3, color=mcell_color)

                ##############
        # ODE SOLVER #
        ##############
        if odesolve == True:
            if i != 5:  # Ca variance is negative?
                ode_results.y[i] = np.where(ode_results.y[i] < 0, 0, ode_results.y[i])
                ode_sd = np.sqrt(np.multiply(ode_results.y[i, :], (1 - ode_results.y[i, :])) / N_CHANS)

                # Cloud plot for sd
                if incl_std == True:
                    a.fill_between(ode_results.t, np.add(ode_results.y[i, :], ode_sd),
                                   np.subtract(ode_results.y[i, :], ode_sd), alpha=0.5, color='lightgrey')

            a.plot(ode_results.t, ode_results.y[i], color='black')

            ##########
    # MARKOV #
    ##########
    if markov == True:
        if incl_std == False:
            for a, n in zip(ax.flatten(), names):
                sns.lineplot(x=avg_markov_results.index, y=n, data=avg_markov_results, linewidth=1, ax=a)
        else:
            for a, n in zip(ax.flatten(), names):
                sns.lineplot(x="timepoints", y=n, data=markov_results, ci="sd", linewidth=1, ax=a)

    # EULER
    # if euler == True:
    #    ax[0,0].plot(trange, vdcc_c0, color='g')
    #    ax[0,1].plot(trange, vdcc_c1, color='g')
    #    ax[1,0].plot(trange, vdcc_c2, color='g')
    #    ax[1,1].plot(trange, vdcc_c3, color='g')
    #    ax[2,0].plot(trange, vdcc_o, color='g')
    #    ax[2,1].plot(trange, ca, color='g')
    #

    plt.show()

vdcc_plot(odesolve=True, euler=False, mcell=True, markov=True, incl_std=True)