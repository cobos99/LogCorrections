import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import namedtuple


ProbConf = namedtuple('ProbConf', ['conf', 'prob'])

pi = np.pi
epsilon = 1e-12

# Kinetic coupling sign
J = -1

#------------------------------------------------------------
# Dispersion relations and Fermi sea
#------------------------------------------------------------
def dispersion_func(L):
    """Dispersion relation for free fermions (antiferro, no minus sign)"""
    def dispersion(k):
        return 2 * J * np.cos( 2 * pi * k / L)
    return dispersion


def wavenumbers(L, apbc=False):
    """Return the wavenumbers `k` for size `L`, with periodic or antiperiodic bc's"""
    offset = 1/2 if apbc else 0
    return [k + offset for k in range(1, L+1)]


def fermi_sea(L, apbc=None):
    """Return a list of wavenumbers that fills the Fermi sea"""
    if apbc is None:
        apbc = default_apbc(L)
    dispersion = dispersion_func(L)
    sea = [ k for k in wavenumbers(L, apbc) if dispersion(k) <= epsilon]
    return sea


def ground_state_energy(L, apbc=False):
    sea = fermi_sea(L, apbc)
    dispersion = dispersion_func(L)
    return sum(dispersion(k) for k in sea)


def fermi_wavenumber(L, apbc=False):
    """Supposedly the Fermi wavenumber"""
    # TODO: works only for J < 0
    ks = wavenumbers(L, apbc)[:L//2]
    dispersion = dispersion_func(L)
    return [ k for k in ks if dispersion(k) <= epsilon][-1]


#------------------------------------------------------------
# Slater and Vandermonde determinants
#------------------------------------------------------------
def vandedet_func(L):
    """Return function that enter the Vandermonde determinant"""
    def f(r1, r2):
        return 4 * (np.sin(pi * (r1 - r2) / L) ** 2)
    return f


def conf_set(L, apbc=False):
    """Return a staggered configuration of the positions of the fermions"""
    l = (L-1)/2
    if l % 2 == 1:
        return tuple(n for n in range(1, L+1, 2))
    else:
        return tuple(n for n in range(2, L+1, 2))


def staggered_conf(L, Nparticles=None, start=1):
    if not Nparticles:
        Nparticles = L // 2
    return tuple(start + 2*n for n in range(0, Nparticles))


def all_confs(L, Nparticles = None, apbc = False):
    """All possible configurations of `Nparticles` fermions on a chain size `L`"""
    if Nparticles is None:
        Nparticles = len(fermi_sea(L, apbc))
    return combinations(range(1, L+1), Nparticles)


def default_apbc(L):
    """ The correct boundary conditions (`apbc` True or False) in order to obtain
    the ground state of the XX chain
    """
    if L % 2 == 0:
        # In correspondence with the XX model:
        #   - For even size, the correct numbers of particles is L/2
        #   - For even num of particles assume APBC
        # L/2 even <=> L mod 4 = 0
        # L/2 odd  <=> L mod 4 = 2
        apbc = True if L % 4 == 0 else False
    else:
        # For odd sizes, the PBC and APBC sectors are degenerate
        # So assume PBC
        apbc = False
    return apbc


def slater_det(L, configuration = None, apbc = None):
    """Compute the Slater determinant directly"""
    if apbc is None:
        apbc = default_apbc(L)
    sea = fermi_sea(L, apbc)
    if not configuration:
        configuration = staggered_conf(L, Nparticles=len(sea))
    else:
        if len(sea) != len(configuration):
            raise ValueError(f"Number of particles ({len(configuration)})"
                             f" does not match the number of wavenumbers k's ({len(sea)})")
    Nparticles = len(configuration)
    mat = np.zeros((Nparticles, Nparticles), dtype=np.complex128)
    for row, r in enumerate(configuration):
        for col, k in enumerate(sea):
            mat[row][col] =  np.exp(-2j * pi * r * k / L)
    return np.abs(np.linalg.det(mat))**2 / (L**Nparticles)


def vandedet(L, positions = None):
    """Compute Slater determinant via Vandermonde determinant"""
    if positions is None:
        positions = conf_set(L)
    # print(f"L = {L}, positions = {positions}")
    f = vandedet_func(L)
    res = 1
    for n, r1 in enumerate(positions):
        for r2 in positions[n+1:]:
            res = res * f(r1, r2)
    return res / (L**len(positions))


def probabilities(L, Nparticles = None, apbc = None) -> list[ProbConf]:
    """Compute the probabilities of the configurations of fermions at size `L`"""
    # TODO calcolare prima quali condizioni  minore energia
    return [
        ProbConf(conf=conf, prob=slater_det(L, conf, apbc))
        for conf in all_confs(L, Nparticles, apbc)
    ]


def max_probs(L, Nparticles = None, apbc=False):
    probs = probabilities(L, Nparticles, apbc)
    maxp = max(probs, key=lambda x: x.prob)
    # Most probabibly there is more than just one max prob conf
    return [
        p for p in probs if np.abs(p.prob - maxp.prob) < 1e-12
    ]


def renyi_inf_entropy(L, apbc=None, pedantic=False):
    print(f" > computing renyi inf entropy for L = {L}:", end=" ")
    if pedantic:
        rie = -np.log(max(probabilities(L, apbc=apbc), key=lambda x: x.prob).prob)
    else:
        rie = -np.log(slater_det(L, apbc=apbc))
    print(f"{rie:.8f}")
    return rie


def shannon_entropy(L):
    """Compute the Shannon entropy for free fermions at size `L`"""
    print(f" * Generating confs for L = {L}")
    confs = all_confs(L)
    print(" * Computing the probabilities")
    probs = np.array([vandedet(L, conf) for conf in confs])
    ent = np.sum([ -p * np.log(p) for p in probs])
    print(f">> Shannon entropy at L = {L}: {ent}")
    return ent


#------------------------------------------------------------
# Some testing
#------------------------------------------------------------
def test_fermi_sea(L):
    print(f"Size L = {L}, Fermi sea for:")
    apbc_sea = fermi_sea(L, apbc=True)
    pbc_sea  = fermi_sea(L, apbc=False)
    print(f"  APBC (size={len(apbc_sea)}): {apbc_sea}")
    print(f"  PBC  (size={len(pbc_sea)}): {pbc_sea}")


def test_gs_energy(L):
    print(f"Size L = {L}, ground state energy for:")
    apbc_sea = fermi_sea(L, apbc=True)
    pbc_sea  = fermi_sea(L, apbc=False)
    print(f"  APBC (fermi sea size={len(apbc_sea)}): {ground_state_energy(L, False)}")
    print(f"  PBC  (fermi sea size={len(pbc_sea)}): {ground_state_energy(L, True)}")


def test_fermi_sea_size(L):
    if L % 2 == 1:
        l = (L-1)/2
        Nparticles = l+1 if l % 2 == 1 else l
    else:
        Nparticles = L/2
    Nparticles = int(Nparticles)
    sea = fermi_sea(L)
    print(f"L = {L}, Nparticles = {Nparticles}, size of Fermi sea = {len(sea)}")
    return Nparticles == len(sea)


def test_fermi_wavenumber(L, apbc=False):
    # TODO il test funziona solo per J < 0
    k_fermi = fermi_wavenumber(L, apbc)
    e = dispersion_func(L)
    print(f"L = {L}, k_fermi = {k_fermi}, e(k_fermi) = {e(k_fermi)}, e(k_fermi + 1) = {e(k_fermi + 1)}")
    return e(k_fermi) < epsilon and e(k_fermi + 1) > epsilon


def test_conf_set(L1, L2, step=1):
    for L in range(L1, L2+1, step):
        print(f"L = {L}, conf = {conf_set(L)}")


def plot_dispersion(L, apbc=False):
    plt.figure()
    ks = wavenumbers(L, apbc)
    sea = fermi_sea(L, apbc)
    momentum = [2 * pi * k / L for k in ks]
    sea_momentum = [2 * pi * k / L for k in sea]
    dispersion = dispersion_func(L);
    gs_energy = ground_state_energy(L, apbc)
    plt.plot(momentum,     [dispersion(k) for k in ks], 'o-k', label=r'$\epsilon(k)$')
    plt.plot(sea_momentum, [dispersion(k) for k in sea], 'd-b', label=f'fermi sea (#occ {len(sea)})')
    plt.plot((0, 2 * pi), (0, 0), ':r')
    plt.legend()
    plt.xlabel(r'$2 \pi k / L$')
    plt.ylabel(r'$\epsilon(k)$')
    plt.title(f"Dispersion relation for L = {L},"
        f" {'anti-periodic' if apbc else 'periodic'} bc, "
        f"$E_0$ = {gs_energy:.4f}")
