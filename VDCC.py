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

# load action potential time series
filename = '~/projects/mcell/simple_geom/input_waveform_data/pre_ap_voltage.txt'
v_m_df = pd.read_csv(filename, delim_whitespace=True, header=None)
v_m = v_m_df.to_numpy()

# interpolate for higher granularity
vm_cubic_interp = interp1d(v_m[:,0]*1000, v_m[:,1], kind = 'cubic',fill_value="extrapolate")
v_m = vm_cubic_interp

# transition rate functions
# from Bischofberger and Jonas (2002) and Nadkarni et al (2010)
# All in units of ms
# Temperature adjustment from paper
q10 = 2.0  # rxn rate increase with 10 degC temp change; assumuption of uniform q10 for all reactions here
delta_temp = 10.0  # parameters are given at 24C so raise temp by 10C to 34C (rat internal temp)
temp_comp = q10 ** (delta_temp / 10.)  # change in rxn rate due to deltaT eqtn

# 1st transition
v1 = 49.14  # mV
def alpha_1(t):
    '''
    alpha 1 forward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for alpha_1
    '''

    a1o = 4.04  # msec-1
    #adjust = 0.5  # changes by Tom and Suhita
    adjust = 1

    return adjust * temp_comp * a1o * np.exp(v_m(t) / v1)


def beta_1(t):
    '''
    beta 1 backward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for beta_1
    '''

    b1o = 2.88  # msec-1
    #adjust = 0.5
    adjust = 1

    return adjust * temp_comp * b1o * np.exp(-v_m(t) / v1)

# 2nd transition
v2 = 42.08  # mV
def alpha_2(t):
    '''
    alpha 2 forward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for alpha_2
    '''

    a2o = 6.70  # msec-1
    #adjust = 0.5  # changes by Tom and Suhita
    adjust = 1

    return adjust * temp_comp * a2o * np.exp(v_m(t) / v2)

def beta_2(t):
    '''
    beta 2 backward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for beta_2
    '''

    b2o = 6.30  # msec-1
    #adjust = 0.5
    adjust = 1

    return adjust * temp_comp * b2o * np.exp(-v_m(t) / v2)

# Transition between 3rd and 4th states
v3 = 55.31  # mV
def alpha_3(t):
    '''
    alpha 3 forward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for alpha_3
    '''


    a3o = 4.39  # msec-1
    #adjust = 0.5  # changes by Tom and Suhita
    adjust = 1

    return adjust * temp_comp * a3o * np.exp(v_m(t) / v3)

def beta_3(t):
    '''
    beta 3 backward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for beta_3
    '''

    b3o = 8.16  # msec-1
    #adjust = 0.5
    adjust = 1

    return adjust * temp_comp * b3o * np.exp(-v_m(t) / v3)

# Transitions between 4th and open state
v4 = 26.55  # mV
def alpha_4(t):
    '''
    alpha 4 forward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for alpha_4
    '''

    a4o = 17.33  # msec-1
    #adjust = 0.5  # changes by Tom and Suhita
    adjust = 1

    return adjust * temp_comp * a4o * np.exp(v_m(t) / v4)


def beta_4(t):
    '''
    beta 4 backward transition rate
    parameters:     time (t)      int/double value or array
    returns:        equation for beta_4
    '''

    b4o = 1.84  # msec-1
    #adjust = 0.5
    adjust = 1

    return adjust * temp_comp * b4o * np.exp(-v_m(t) / v4)

# Current influx rate constant
# 1/ms
def k_ca(t):
    f_ca = 511080.19/1802251 # correction factor for only calcium in current
    g_hva = f_ca * 1.55 * 2.4e-12  # conductance (3.72 pS; adjusted for temperature)
    c = 80.36  # mV
    d = 0.3933  # parameter determining current rectification and reversal potential
    e_c = 1.6e-19  # Coulombie charge of one electron (e = F/NA)
    Z_ca = 2  # Charge of calcium ion
    #adjust = 0.183  # changes by Tom and Suhita
    adjust = 1

    return adjust * g_hva * v_m(t) * (d - np.exp(-v_m(t) / c)) / ((1000 ** 2) * Z_ca * e_c * (1 - np.exp(v_m(t) / c)))

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