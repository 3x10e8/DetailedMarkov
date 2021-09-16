import numpy as np
from scipy.interpolate import interp1d
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
import os


def stimulus(fname):
    '''
    Interpolates and creates function for stimulus input from file
    :param fname: stimulus input
    :return: voltage function
    '''

    v_m = np.loadtxt(fname)
    v_m = interp1d(v_m[:, 0] * 1000, v_m[:, 1], kind='cubic',
                   fill_value="extrapolate")

    return v_m

os.chdir('/Users/margotwagner/projects/DetailedMarkov/scripts')
fname = './pre_ap_voltage.txt'
v_m = stimulus(fname)

# transition rate functions
# from Bischofberger and Jonas (2002) and Nadkarni et al (2010/2012)
# All in units of ms
# Temperature adjustment from paper
# rxn rate increase with 10 degC temp change; assumuption of uniform
q10 = 2.0
delta_temp = 10.0  # raise temp from 24C to 34C
temp_comp = q10 ** (delta_temp / 10.)  # change T

# 1ST TRANSITION
v1 = 49.14  # mV

def a1(t):
    '''
    alpha 1 forward transition rate
    :param t: time
    :return: equation for alpha_1
    '''

    a1o = 4.04  # msec-1

    return temp_comp * a1o * np.exp(v_m(t) / v1)

def b1(t):
    '''
    beta 1 backward transition rate
    :param t: time int/double value or array
    :return: equation for beta_1
    '''

    b1o = 2.88  # msec-1

    return temp_comp * b1o * np.exp(-v_m(t) / v1)

# 2ND TRANSITION
v2 = 42.08  # mV

def a2(t):
    '''
    alpha 2 forward transition rate
    :param t: time
    :return: equation for alpha_2
    '''

    a2o = 6.70  # msec-1

    return temp_comp * a2o * np.exp(v_m(t) / v2)

def b2(t):
    '''
    beta 2 backward transition rate
    :param t: time
    :return: equation for beta_2
    '''

    b2o = 6.30  # msec-1

    return temp_comp * b2o * np.exp(-v_m(t) / v2)

# 3RD TRANSITION
v3 = 55.31  # mV

def a3(t):
    '''
    alpha 3 forward transition rate
    :param t: time
    :return: equation for alpha_3
    '''

    a3o = 4.39  # msec-1

    return temp_comp * a3o * np.exp(v_m(t) / v3)

def b3(t):
    '''
    beta 3 backward transition rate
    :param t: time
    :return: equation for beta_3
    '''

    b3o = 8.16  # msec-1

    return temp_comp * b3o * np.exp(-v_m(t) / v3)

# 4TH TRANSITION
v4 = 26.55  # mV

def a4(t):
    '''
    alpha 4 forward transition rate
    :param t: time
    :return: equation for alpha_4
    '''

    a4o = 17.33  # msec-1

    return temp_comp * a4o * np.exp(v_m(t) / v4)

def b4(t):
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
    g_hva = f_ca * 1.55 * 2.4e-12  # conductance (3.72 pS; adj for temp)
    c = 80.36   # mV
    d = 0.3933  # parameter determining I rectification and reversal V
    e_c = 1.6e-19  # Coulombic charge of one electron (e = F/NA)
    Z_ca = 2  # Charge of calcium ion

    return g_hva * v_m(t) * (d - np.exp(-v_m(t) / c)) / \
           ((1000 ** 2) * Z_ca * e_c * (1 - np.exp(v_m(t) / c)))


##########################
#### SDE INTEGRATION #####
##########################

# MAKE THIS BETTER!!!
def sde(y0, t_start, t_stop, dt):
    t = np.arange(t_start, t_stop + dt, dt)  # range of t values

    # Initialize states
    c0, c1, c2, c3, op, ca = [np.zeros(len(t)) for _ in range(len(y0))]
    c0s, c1s, c2s, c3s, ops, cas = [np.zeros(len(t)) for _ in range(len(y0))]

    nc = y0[0]  # number of channels

    # Initial conditions
    c0[0] = y0[0] / nc
    c1[0] = y0[1]
    c2[0] = y0[2]
    c3[0] = y0[3]
    op[0] = y0[4]
    ca[0] = y0[5]

    c0s[0] = y0[0] / nc
    c1s[0] = y0[1]
    c2s[0] = y0[2]
    c3s[0] = y0[3]
    ops[0] = y0[4]
    cas[0] = y0[5]

    # Simulate
    for n in range(len(t) - 1):
        # Add noise to current step
        # truncate (set to zero where <0, set to 1 where >1)

        c0[n] = np.clip(c0[n], 0, 1)
        c1[n] = np.clip(c1[n], 0, 1)
        c2[n] = np.clip(c2[n], 0, 1)
        c3[n] = np.clip(c3[n], 0, 1)
        op[n] = np.clip(op[n], 0, 1)

        with np.errstate(all='raise'):
            try:
                c0s[n] = normal(c0[n], np.sqrt(c0[n] * (1 - c0[n]) / nc))
                c1s[n] = normal(c1[n], np.sqrt(c1[n] * (1 - c1[n]) / nc))
                c2s[n] = normal(c2[n], np.sqrt(c2[n] * (1 - c2[n]) / nc))
                c3s[n] = normal(c3[n], np.sqrt(c3[n] * (1 - c3[n]) / nc))
                ops[n] = normal(op[n], np.sqrt(op[n] * (1 - op[n]) / nc))
            # cas[n] = normal(ca[n], np.sqrt(ca[n] / nc)) # unclear
            except FloatingPointError:
                print("warning")
                print("c0:", c0[n], c0s[n - 1])
                print("c1:", c1[n], c1s[n - 1])
                print("c2:", c2[n], c2[n - 1], c2s[n - 1])
                print("c3:", c3[n], c3s[n - 1])
                print("op:", op[n], op[n - 1], ops[n - 1])
                break

        # print(c3[n])

        # truncate (set to zero where <0, set to 1 where >1)
        c0s[n] = np.clip(c0s[n], 0, 1)
        c1s[n] = np.clip(c1s[n], 0, 1)
        c2s[n] = np.clip(c2s[n], 0, 1)
        c3s[n] = np.clip(c3s[n], 0, 1)
        ops[n] = np.clip(ops[n], 0, 1)
        # cas[n] = np.clip(cas[n], 0, 1)

        # print(c3[n])

        # Solve for next val for mean sol (prev mean step + deriv using noisy sol)
        c0[n + 1] = c0[n] + (b1(t[n]) * c1s[n] - a1(t[n]) * c0s[n]) * dt

        c1[n + 1] = c1[n] + (a1(t[n]) * c0s[n] + b2(t[n]) * c2s[n]
                             - (b1(t[n]) + a2(t[n])) * c1s[n]) * dt

        c2[n + 1] = c2[n] + (a2(t[n]) * c1s[n] + b3(t[n]) * c3s[n]
                             - (b2(t[n]) + a3(t[n])) * c2s[n]) * dt

        c3[n + 1] = c3[n] + (a3(t[n]) * c2s[n] + b4(t[n]) * ops[n]
                             - (b3(t[n]) + a4(t[n])) * c3s[n]) * dt

        op[n + 1] = op[n] + (a4(t[n]) * c3s[n] - b4(t[n]) * ops[n]) * dt

        # ca[n+1] = ca[n] + (k_ca(t[n])*ops[n]) * dt

    return c0, c1, c2, c3, op, ca, c0s, c1s, c2s, c3s, ops, cas


# Run simulation
# Initial conditions
y0 = [65, 0, 0, 0, 0, 0]

# Time
dt = 1e-3
t_start = 0
t_stop = 10
t = np.arange(t_start, t_stop + dt, dt)  # range of t values

print("Starting simulation...")
c0, c1, c2, c3, op, ca, c0s, c1s, c2s, c3s, ops, cas = sde(y0, t_start, t_stop, dt)

print("Done.")
plt.plot(t, ops)
plt.plot(t, op)
plt.xlim(0, 6)

plt.savefig('/Users/margotwagner/projects/DetailedMarkov/scripts/sde.png')
plt.show()