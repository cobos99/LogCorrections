import shannon as snn
import matplotlib.pyplot as plt
from numpy import pi

#------------------------------------------------------------
# Some testing
#------------------------------------------------------------
def test_fermi_sea(L):
    print(f"Size L = {L}, Fermi sea for:")
    apbc_sea = snn.fermi_sea(L, apbc=True)
    pbc_sea  = snn.fermi_sea(L, apbc=False)
    print(f"  APBC (size={len(apbc_sea)}): {apbc_sea}")
    print(f"  PBC  (size={len(pbc_sea)}): {pbc_sea}")
    correct_apbc = snn.default_apbc(L)
    print(f"  Correct BC: {'apbc' if correct_apbc else 'pbc'}")


def test_gs_energy(L):
    print(f"Size L = {L}, ground state energy for:")
    apbc_sea = snn.fermi_sea(L, apbc=True)
    pbc_sea  = snn.fermi_sea(L, apbc=False)
    print(f"  APBC (fermi sea size={len(apbc_sea)}): {snn.ground_state_energy(L, False)}")
    print(f"  PBC  (fermi sea size={len(pbc_sea)}): {snn.ground_state_energy(L, True)}")
    correct_apbc = snn.default_apbc(L)
    print(f"  Correct BC: {'apbc' if correct_apbc else 'pbc'}")


def test_fermi_sea_size(L):
    if L % 2 == 1:
        l = (L-1)/2
        Nparticles = l+1 if l % 2 == 1 else l
    else:
        Nparticles = L/2
    Nparticles = int(Nparticles)
    sea = snn.fermi_sea(L)
    print(f"L = {L}, Nparticles = {Nparticles}, size of Fermi sea = {len(sea)}")
    return Nparticles == len(sea)


def test_fermi_wavenumber(L, apbc=False):
    k_fermi = snn.fermi_wavenumber(L, apbc)
    e = snn.dispersion_func(L)
    if snn.J < 0:
        print(f"L = {L}, k_fermi = {k_fermi}, e(k_fermi) = {e(k_fermi)}, e(k_fermi + 1) = {e(k_fermi + 1)}")
    else:
        print(f"L = {L}, k_fermi = {k_fermi}, e(k_fermi) = {e(k_fermi)}, e(k_fermi - 1) = {e(k_fermi - 1)}")
    return e(k_fermi) < snn.epsilon and e(k_fermi + 1) > snn.epsilon


def test_staggered_conf(L1, L2, step=1):
    for L in range(L1, L2+1, step):
        print(f"L = {L}, conf = {snn.staggered_conf(L)}")


def plot_dispersion(L, apbc=False):
    plt.figure()
    ks = snn.wavenumbers(L, apbc)
    sea = snn.fermi_sea(L, apbc)
    momentum = [2 * pi * k / L for k in ks]
    sea_momentum = [2 * pi * k / L for k in sea]
    dispersion = snn.dispersion_func(L);
    gs_energy = snn.ground_state_energy(L, apbc)
    plt.plot(momentum,     [dispersion(k) for k in ks], 'o-k', label=r'$\epsilon(k)$')
    plt.plot(sea_momentum, [dispersion(k) for k in sea], 'd-b', label=f'fermi sea (#occ {len(sea)})')
    plt.plot((0, 2 * pi), (0, 0), ':r')
    plt.legend()
    plt.xlabel(r'$2 \pi k / L$')
    plt.ylabel(r'$\epsilon(k)$')
    plt.title(f"Dispersion relation for L = {L},"
        f" {'anti-periodic' if apbc else 'periodic'} bc, "
        f"$E_0$ = {gs_energy:.4f}")
