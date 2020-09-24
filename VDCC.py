'''Script for looking at VDCC transitions and calcium influx based on equations
from Bischofberger and Jone (2002). Similar to script used for Nadkarni pub.
Temperature is adjusted for physiologically relevant temperature. IV curve
adjusted according to Nadkarni (to account for only calcium influx)'''


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

    v_m_df = pd.read_csv(fname, delim_whitespace=True, header=None)
    v_m = v_m_df.to_numpy()

    # interpolate for higher granularity
    vm_cubic_interp = interp1d(v_m[:, 0] * 1000, v_m[:, 1], kind='cubic', fill_value="extrapolate")
    v_m = vm_cubic_interp

    return v_m


fname = '~/projects/mcell/simple_geom/input_waveform_data/pre_ap_voltage.txt'
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


# PLOT TRANSITION RATES
"""
trange = np.arange(0, 11, 1e-3)

# Plot transition rates
fig, ax = plt.subplots(1, 2, figsize = (15,5))

ax[0].plot(trange,v_m(trange))
ax[0].set_title('Membrane Voltage (Vm)')
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Voltage (mV)')


ax[1].plot(trange,k_ca(trange))
ax[1].set_title('Calcium influx rate constant, $k_{ca}$')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Transition rate (1/ms)')
plt.show()

fig, ax = plt.subplots(2, 4, figsize=(17, 7))

for i in range(2):
    for j in range(4):
        ax[i][j].set_xlabel('Time (ms)')
        ax[i][j].set_ylabel('Transition rate')
        if i == 0:
            ax[i][j].set_title(r'VDCC transition, $\alpha_{%s}$' % (j + 1))
            if j == 0:
                ax[i][j].plot(trange, alpha_1(trange))
            elif j == 1:
                ax[i][j].plot(trange, alpha_2(trange))
            elif j == 2:
                ax[i][j].plot(trange, alpha_3(trange))
            elif j == 3:
                ax[i][j].plot(trange, alpha_4(trange))
        else:
            ax[i][j].set_title(r'VDCC transition, $\beta_{%s}$' % (j + 1))
            if j == 0:
                ax[i][j].plot(trange, beta_1(trange))
            elif j == 1:
                ax[i][j].plot(trange, beta_2(trange))
            elif j == 2:
                ax[i][j].plot(trange, beta_3(trange))
            elif j == 3:
                ax[i][j].plot(trange, beta_4(trange))

fig.tight_layout()
plt.show()
"""



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
##################################
### OR LOAD MARKOV SIMULATION ####
##################################
N_CHANS=65
fname = "markov_1000_runs*.csv"
sep_results_df = (pd.read_csv(file) for file in sorted(glob.glob(fname)))
markov_results = pd.concat(sep_results_df)
markov_results['vdcc_o_norm'] = markov_results['vdcc_o']/N_CHANS

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
ode_results = solve_ivp(vdcc_odes,[t_start, t_stop+dt], [1, 0, 0, 0, 0, 0],
                        t_eval=trange)

##################################
####### MCELL SIMULATION #########
##################################
# LOAD DATA
# Constants
MCELL_DIR = "/Users/margotwagner/projects/mcell/simple_geom/model_1/" \
            "model_1_vdcceqtns_tchange_nodendrite_files/mcell/output_data/" \
            "react_data/"
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
for file_path in glob.glob(os.path.join(MCELL_DIR,
                                        "seed_00001/vdcc_pre_o.World.dat")):
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

# NORMALIZE DATA
vdcc_o_idx = 0

mcell_results = np.copy(mcell_data)
n_chans=65
for seed in range(SEEDS):
    mcell_results[vdcc_o_idx][:, 1, seed] = mcell_results[vdcc_o_idx][:, 1, seed] / (n_chans)


##################################
###### COMPARE SIMULATIONS #######
##################################
plt.figure(figsize=(8,6), dpi=100)
# MARKOV
sns.lineplot(x="timepoints", y="vdcc_o_norm", ci="sd", data=markov_results, linewidth=2)

# ODE
#plt.plot(ode_results.t, ode_results.y[4,:], color='cyan', linestyle = ':', linewidth=2)
#sd = np.sqrt(np.multiply(ode_results.y[4,:], (1 - ode_results.y[4,:]))/N_CHANS)
#plt.plot(ode_results.t, np.add(ode_results.y[4,:], sd), color='darkgrey')
#plt.plot(ode_results.t, np.subtract(ode_results.y[4,:], sd), color='darkgrey')

# MCELL
# Mean
mean = np.mean(mcell_results[0], axis=2)
plt.plot(mean[:,0]*1000, mean[:,1], color='black', label='MCell', )

# SEM
#sem = stats.sem(mcell_data_norm[0], axis=2)
#plt.plot(mean[:,0]*1000, mean[:,1] + sem[:,1], 'C0')
#plt.plot(mean[:,0]*1000, mean[:,1] - sem[:,1], 'C0')

plt.xlim(1.5,5)
plt.show()


