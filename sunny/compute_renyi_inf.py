import numpy as np
import matplotlib.pyplot as plt

from shannon import renyi_inf_entropy as rie
from scipy.optimize import curve_fit

lens_even = np.array(range(4, 101, 2))
lens_odd = np.array(range(5, 102, 2))

lens_avg = (lens_even + lens_odd) / 2


smax_even = np.array([ rie(int(L)) for L in lens_even ])
smax_odd  = np.array([ rie(int(L)) for L in lens_odd  ])

fit_fn = lambda x, a, b, c: a*x + b*np.log(x) + c

fit_res_even = tuple(curve_fit(fit_fn, lens_even, smax_even)[0])
fit_res_odd  = tuple(curve_fit(fit_fn, lens_odd,  smax_odd )[0])

plt.subplots(1, 2)

plt.subplot(1, 2, 1)
plt.title("Shannon-Renyi inf entropy")
plt.plot(lens_even, smax_even, 'o-', label="even")
plt.xlabel(r"$L$")
plt.ylabel(r"$- \log p_{\text{max}}$")

plt.plot(lens_odd, smax_odd, 'o-', label="odd")
plt.ylabel(r"$- \log p_{\text{max}}$")
plt.legend()

plt.grid()

plt.subplot(1, 2, 2)
plt.title("Difference between even and odd")
plt.plot(lens_avg, smax_odd - smax_even, 'o-', label="diff")
plt.grid()
plt.xlabel(r"$L$")
plt.ylabel("difference")

def print_fit_res(fit_res):
    print(f"\t alpha = {fit_res[0]}")
    print(f"\t beta  = {fit_res[1]}")
    print(f"\t gamma = {fit_res[2]}")
    print()

print("Fit results for odd lengths:")
print_fit_res(fit_res_even)
print("Fit results for even lengths:")
print_fit_res(fit_res_odd)
