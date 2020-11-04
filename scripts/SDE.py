import numpy as np
from scipy.interpolate import interp1d


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



##########################
### EULER INTEGRATION ####
##########################

# MAKE THIS BETTER!!!
def sde(y0, t_start, t_stop, dt):
    trange = np.arange(t_start, t_stop + dt, dt)

    # Initialize states
    c0_avg, c1_avg, c2_avg, c3_avg, o_avg, ca_avg = [np.zeros(len(trange)) for _ in range(len(y0))]

    # Initial conditions
    c0_avg[0] = y0[0]
    c1_avg[0] = y0[1]
    c2_avg[0] = y0[2]
    c3_avg[0] = y0[3]
    o_avg[0] = y0[4]
    ca_avg[0] = y0[5]

    # Simulate
    for n in range(len(trange) - 1):
        # Solve for next step for mean solution (prev mean step + deriv using mean)
        c0_avg[n + 1] = c0_avg[n] + (beta_1(trange[n]) * c1_avg[n] - alpha_1(trange[n]) * c0_avg[n]) * dt

        c1_avg[n + 1] = c1_avg[n] + (alpha_1(trange[n]) * c0_avg[n] + beta_2(trange[n]) * c2_avg[n] - (
                                        beta_1(trange[n]) + alpha_2(trange[n])) * c1_avg[n]) * dt

        c2_avg[n + 1] = c2_avg[n] + (alpha_2(trange[n]) * c1_avg[n] + beta_3(trange[n]) * c2_avg[n] - (
                                        beta_2(trange[n]) + alpha_3(trange[n])) * c2_avg[n]) * dt

        c3_avg[n + 1] = c3_avg[n] + (alpha_3(trange[n]) * c2_avg[n] + beta_4(trange[n]) * o_avg[n] - (
                                        beta_3(trange[n]) + alpha_4(trange[n])) * c3_avg[n]) * dt

        o_avg[n + 1] = o_avg[n] + (alpha_4(trange[n]) * c3_avg[n] - beta_4(trange[n]) * o_avg[n]) * dt

        ca_avg[n + 1] = ca_avg[n] + (k_ca(trange[n]) * o_avg[n]) * dt

    return c0_avg, c1_avg, c2_avg, c3_avg, o_avg, ca_avg


# Run stimulation
# Initial conditions
y0 = [1, 0, 0, 0, 0, 0]

# Time
dt = 1e-3
t_start = 0
t_stop = 10


#vdcc_c0, vdcc_c1, vdcc_c2, vdcc_c3, vdcc_o, ca = euler(y0, t_start, t_stop, dt)

