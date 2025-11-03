import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, cos, sin, sqrt

def momenta(L: int, apbc: bool = False):
    if apbc:
        return [ 2 * pi * (k + 1/2) / L for k in range(0, L) ]
    else:
        return [ 2 * pi * k / L for k in range(0, L) ]

def dispersion_fn(h: float, delta: float):
    return lambda p: sqrt( (cos(p) + h)**2 + (delta * sin(p))**2 )

ising_crit_dispersion = dispersion_fn(h=1.0, delta=1.0)
xx_dispersion = dispersion_fn(h=0.0, delta=0)

def crit_gs_energy(L, apbc):
    return -0.5 * sum([ising_crit_dispersion(p) for p in momenta(L, apbc)])

odd_sizes = range(5, 50, 2)
even_sizes = range(6, 50, 2)
energies_pbc_even  = np.array([ crit_gs_energy(L, apbc=False) for L in even_sizes ])
energies_pbc_odd   = np.array([ crit_gs_energy(L, apbc=False) for L in odd_sizes  ])
energies_apbc_even = np.array([ crit_gs_energy(L, apbc=True)  for L in even_sizes ])
energies_apbc_odd  = np.array([ crit_gs_energy(L, apbc=True)  for L in odd_sizes  ])

def greater(x, y):
    return x - y > 1e-5

# for L, E_pbc, E_apbc in zip(sizes, gs_energies_pbc, gs_energies_apbc):
#     print(f"L = {L} \t E pbc = {E_pbc:.4f}\tE apbc = {E_apbc:.4f}", end="\t")
#     if greater(E_pbc, E_apbc):
#         print("apbc lower (L odd)")
#     elif greater(E_apbc, E_pbc):
#         print("pbc lower (L even)")
#     else:
#         print("same")

plt.subplots(1, 2, figsize=(16, 7))
marker_options = dict(markersize=10, markeredgewidth=3)

plt.subplot(1, 2, 1)
plt.plot(even_sizes, energies_pbc_even - energies_apbc_even, "x-", label="odd - even", **marker_options)
plt.legend()
plt.title(r"$L$ even ($E^{\text{even}}$ is lower)")
plt.xlabel("$L$")
plt.ylabel(r"$\Delta E_{\text{gs}}$")
plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)

plt.subplot(1, 2, 2)
plt.plot(odd_sizes, energies_pbc_odd - energies_apbc_odd, "x-", label="odd - even", **marker_options)
plt.legend()
plt.title(r"$L$ odd ($E^{\text{odd}}$ is lower)")
plt.xlabel("$L$")
plt.ylabel(r"$\Delta E_{\text{gs}}$")
plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)

plt.tight_layout()
plt.savefig("graphs/ising_gs_energy_comparison.pdf")

plt.show()
# plt.suptitle("GS energy diff. between the parity sectors of critical Ising")

