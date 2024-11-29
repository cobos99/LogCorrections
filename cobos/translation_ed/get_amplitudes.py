from scipy.optimize import curve_fit
import numpy as np
import amptools

def to_fit_renyi_inf(N, a, b, c):
    return a*N + b*np.log(N) + c

def to_fit_shannon(N, a, b, c):
    return a*N + b*np.log(N) + c

N_odd_range = [4, 22]
global_neg = False
delta_arr = np.linspace(-1, 1, 21)
print_mode = True
initials = [0.1, 0.1, 0.1]

N_arr = np.arange(*N_odd_range, 2)
max_amps_arr = amptools.get_xxz_numerical_max_amps(N_arr, delta_arr, global_neg, print_mode=print_mode)

# %%
global_neg = True
max_amps_arr = amptools.get_xxz_numerical_max_amps(N_arr, delta_arr, global_neg, print_mode=print_mode)