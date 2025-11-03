import numpy as np
from itertools import combinations
from dataclasses import dataclass
from collections.abc import Callable, Iterator

# Class for the probability of a configuration
@dataclass
class ProbConf:
    conf: tuple[int] # the configuration of particles (i.e. their positions)
                     # the position index start from 1
    prob: float      # probability of such configuration

pi = np.pi
epsilon = 1e-12



# Kinetic coupling sign
J = 1

#------------------------------------------------------------
# Dispersion relations and Fermi sea
#------------------------------------------------------------

def dispersion_func(L: int) -> Callable[[int], float]:
    """
    Return the dispersion *function* of free fermions on a chain of size `L`.
    The kinetic coupling is determined by the global variable `J`
    """
    def dispersion(k):
        return 2 * J * np.cos( 2 * pi * k / L)
    return dispersion


def wavenumbers(L: int, apbc: bool = False) -> list[int]:
    """
    Return the list of wavenumbers `k` for size `L`.
    The wavenumbers `k`s are integers, such that `2*pi*k/L` are the momenta.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    """
    offset = 1/2 if apbc else 0
    return [k + offset for k in range(1, L+1)]


def default_apbc(L: int) -> bool:
    """
    Deduce the correct boundary conditions in order to obtain
    the ground state of the XX chain with periodic boundary conditions.
    Returns `True` if antiperiodic BC have to be applied.
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
        apbc = False
    return apbc


def fermi_sea(L: int, apbc: bool | None = None) -> list[int]:
    """
    Return the list of wavenumbers `k`s that fills the Fermi sea
    for a system of size `L`.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    """
    if apbc is None:
        apbc = default_apbc(L)
    dispersion = dispersion_func(L)
    sea = [ k for k in wavenumbers(L, apbc) if dispersion(k) <= epsilon]
    return sea


def ground_state_energy(L: int, apbc: bool | None = None) -> float:
    """
    Returns the ground state energy for size `L`.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    """
    dispersion = dispersion_func(L)
    return sum(dispersion(k) for k in fermi_sea(L, apbc))


def fermi_wavenumber(L: int, apbc: bool | None = None) -> list[int]:
    """
    Returns the Fermi wavenumber `k_F` for a system of size `L`.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    """
    if apbc is None:
        apbc = default_apbc(L)
    momenta = wavenumbers(L, apbc)[:L//2]
    dispersion = dispersion_func(L)
    if J < 0:
        return [ k for k in momenta if dispersion(k) <= epsilon][-1]
    else:
        return [ k for k in momenta if dispersion(k) <= epsilon][0]


#------------------------------------------------------------
# Particle configurations
#------------------------------------------------------------

def staggered_conf(L: int, Np: int | None = None, start: int = 1) -> tuple[int]:
    """
    Return a staggered configuration of particle positions (alternate occupied
    and empty site) on a chain of size `L`. The position indices start from 1.
    If the number of particles `Np` is not specified, then `L/2` is assumed.
    `start` marks the first occupied site, by default is `1`.
    """
    if Np is None:
        Np = L // 2
    if start + 2*(Np - 1) > L:
        raise ValueError(f"The values start={start} and Np={Np} exceeds the size of chain L={L}")
    return tuple(start + 2*n for n in range(Np))


def all_confs(L: int, Np: int | None = None, apbc: bool | None = None) -> Iterator[tuple[int]]:
    """
    All possible configurations of `Np` particles on a chain size `L`.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    The position indices start from 1.
    """
    if Np is None:
        Np = len(fermi_sea(L, apbc))
    return combinations(range(1, L+1), Np)


#------------------------------------------------------------
# Slater determinants
#------------------------------------------------------------

def slater_det(
        L: int,
        apbc: bool | None = None,
        Np: int | None = None,
        conf: tuple[int] | None = None
    ) -> float:
    """
    Compute the Slater determinant directly of the given configuration `conf`.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.
    """
    # Check if Np is a correct value
    if apbc is None:
        apbc = default_apbc(L)
    sea = []
    # This fucking sucks
    if Np is not None:
        if L % 2 == 1:
            if L != 2*Np + 1 and L != 2*Np - 1:
                raise ValueError(f"Invalid value of Np = {Np} for L = {L}")
            sea = fermi_sea(L, apbc)
            # This is very ugly
            if len(sea) != Np:
                apbc = not apbc
                sea = fermi_sea(L, apbc)
        else:
            if L != 2*Np:
                raise ValueError(f"Invalid value of Np = {Np} for L = {L}")
            sea = fermi_sea(L, apbc)
        # Specifying Np overwrite conf
        conf = staggered_conf(L, Np=Np)
    else:
        sea = fermi_sea(L, apbc)
        if not conf:
            conf = staggered_conf(L, Np=len(sea))
        if len(sea) != len(conf):
            raise ValueError(f"Number of particles in `conf` ({len(conf)}) does not"
                             f" match the number of occupied modes in the Fermi sea ({len(sea)})")
        Np = len(conf)

    Nparticles = len(conf)
    positions = np.array(conf).reshape((Nparticles, 1)) # row vector
    momenta = np.array(sea)
    slater_mat = np.exp(-2j * pi * positions * momenta / L)
    return np.abs(np.linalg.det(slater_mat))**2 / (L**Nparticles)


#------------------------------------------------------------
# Vandermonde determinants
#------------------------------------------------------------

def vandedet_func(L: int) -> Callable[[int, int], float]:
    """
    Return the main function that enter in the Vandermonde determinant
    `L` is the size of the chain.
    """
    def f(r1, r2):
        return 4 * (np.sin(pi * (r2 - r1) / L) ** 2)
    return f


def vandemonde_det(L: int, conf: tuple[int] | None = None) -> float:
    """
    Compute the Vandermonde determinant for the given configuration `conf`.
    """
    # TODO not finished and not tested
    if conf is None:
        conf = staggered_conf(L, Np=len(fermi_sea(L)))
    f = vandedet_func(L)
    res = 1
    for n, r1 in enumerate(conf):
        for r2 in conf[n+1:]:
            res = res * f(r1, r2)
    return res / (L**len(conf))


#------------------------------------------------------------
# Probabilities of configurations
#------------------------------------------------------------
def probabilities(
        L: int,
        Np: int | None = None,
        apbc: bool | None = None,
        method: str = "slater"
    ) -> Iterator[ProbConf]:
    """
    Compute the probabilities of all possible configurations of `Np` particles
    on a chain of size `L`.
    If not specified, `Np` is the numbers of occupied modes in the Fermi sea.
    Antiperiodic BC is automatically deduced, unless specified by `apbc`.

    The probabilities can be computed either directly with the Slater
    determinant or with the Vandermonde determinant, by setting the `method`
    parameter to "slater" or "vande" resp.

    The function returns a generator, not a list.
    In this way we avoid storing in memory all the possible configurations.
    The number of combinations grows exponentially.
    """
    if method == "slater":
        func = slater_det
    elif method == "vande":
        func = vandemonde_det
    else:
        raise ValueError(f"Value '{method}' of `method` not recognized")

    return (
        ProbConf(conf=conf, prob=func(L, conf, apbc))
        for conf in all_confs(L, Np, apbc)
    )


def max_probs(L: int, Np: int | None = None, apbc: bool | None=None):
    """
    Return a list of the configurations with maximum probabilities
    """
    probs = list(probabilities(L, Np, apbc))
    maxp = max(probs, key=lambda x: x.prob)
    # Most probabibly there is more than just one max prob conf
    return [
        p for p in probs
            if np.abs(p.prob - maxp.prob) < epsilon
    ]


def max_probs2(L, Np : bool | None = None, apbc: bool | None = None):
    """
    Return a list of the configurations with maximum probabilities.
    Alternative method.
    """
    maxp = -1.0
    max_confs = []
    for p in probabilities(L, Np, apbc):
        if p.prob - maxp > epsilon:
            # new maximum
            maxp = p.prob
            max_confs = [p]
        elif abs(p.prob - maxp) < epsilon:
            # same probability
            max_confs.append(p)
    return max_confs



#------------------------------------------------------------
# Renyi inf entropy and Shannon entropy
#------------------------------------------------------------
def renyi_inf_entropy(L: int, apbc: bool | None = None, pedantic: bool =False, Np: None | int = None):
    """
    Compute the Renyi entropy of infinite order, i.e. -log(p_max), for a system
    of size `L`.
    With the option `pedantic`, search for the actual configuration with maximum
    probability instead of supposing that is the one with a staggered
    configuration.
    """
    # print(f" > computing renyi inf entropy for L = {L}:", end=" ")
    if pedantic:
        rie = -np.log(max(probabilities(L, apbc=apbc), key=lambda x: x.prob).prob)
    else:
        if Np is not None:
            rie = -np.log(slater_det(L, Np=Np))
        else:
            rie = -np.log(slater_det(L, apbc=apbc))
    # print(f"{rie:.8f}")
    return rie


def shannon_entropy(L: int, apbc: bool | None = None):
    """
    Compute the Shannon entropy for free fermions at size `L`
    """
    print(f" > Computing the probabilities for L = {L}")
    confs = all_confs(L)
    probs = np.array([slater_det(L, conf) for conf in confs])
    ent = np.sum([ -p * np.log(p) for p in probs])
    print(f" >>> Shannon entropy at L = {L}: {ent}")
    return ent
