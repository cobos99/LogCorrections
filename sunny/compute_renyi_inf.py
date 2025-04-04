import numpy as np
import matplotlib.pyplot as plt

from shannon import renyi_inf_entropy as rie
from scipy.optimize import curve_fit

lens_even = np.array(range(4, 51, 2))
lens_odd = np.array(range(5, 52, 2))


smax_even = np.array([ rie(int(L)) for L in lens_even ])
smax_odd  = np.array([ rie(int(L)) for L in lens_odd  ])

fit_fn = lambda x, a, b, c: a*x + b*np.log(x) + c

fit_res_even = tuple(curve_fit(fit_fn, lens_even, smax_even)[0])
fit_res_odd  = tuple(curve_fit(fit_fn, lens_odd,  smax_odd )[0])

def print_fit_res(fit_res):
    print(f"\t alpha = {fit_res[0]}")
    print(f"\t beta  = {fit_res[1]}")
    print(f"\t gamma = {fit_res[2]}")
    print()


marker_options = dict(markersize=20, markeredgewidth=3)

# Matplotlib settings
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{physics} \usepackage{mathtools}")
plt.rc("font", family="serif", size=28, weight="bold")

plt.subplots(1, 2, figsize=(15, 7))

# Smax entropy
plt.subplot(1, 2, 1)
plt.title(r"$S_{\infty}$ entropy for $\Delta = 0$")
plt.plot(lens_even, smax_even, '.', label=r"$L = 2 N$", **marker_options)
plt.plot(lens_odd,  smax_odd,  '.', label=r"$L = 2 N + 1$", **marker_options)

plt.xlabel(r"$L$")
plt.ylabel(r"$- \log \max_i p_i$")
plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)
plt.tick_params(width=2, length=10, direction="in")
plt.legend(shadow=True)
plt.legend()

# plt.grid()

plt.subplot(1, 2, 2)
plt.title(r"$S_{\infty}$ difference")
plt.plot(lens_even / 2, smax_odd - smax_even, '+', label="Numerical", **marker_options)
diff = 0.25 * np.log(lens_even / 2) + fit_res_even[0] + fit_res_odd[2] + fit_res_odd[1] * np.log(2) + 0.02
plt.plot(
    lens_even / 2,
    diff,
     "--",
    label=r"$\frac{1}{4} \log \ell + \text{const}$",
    linewidth=3
)
# plt.grid()
plt.xlabel(r"$\ell$")
plt.ylabel(r"$S_{\infty}^{\text{odd}} - S_{\infty}^{\text{even}}$")
plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)
plt.tick_params(width=2, length=10, direction="in")
plt.legend(shadow=True, loc="lower right")

# Save fig
plt.tight_layout()
plt.savefig("Smax_diff.pdf", bbox_inches='tight')

print("Fit results for even lengths:")
print_fit_res(fit_res_even)
print("Fit results for odd lengths:")
print_fit_res(fit_res_odd)

plt.savefig("max_entropy_XX.pdf", bbox_inches="tight")
plt.show()
