import numpy as np
import pandas as pd

from numpy import pi
from itertools import combinations


def amplitude(L: int, alpha: float, conf: np.array(int)) -> float:
    z = np.exp(2j * pi / L)
    # c = np.array(conf, dtype=np.int32)
    N = len(conf)
    conf_T = conf.reshape((N, 1))
    mat = z**conf_T - z**conf
    return np.prod(mat + np.eye(N))**(2 * alpha)


def imps_normalization(L: int, N: int, alpha: float) -> float:
    return sum(
                np.abs(amplitude(L, alpha, np.array(conf)))**2
                    for conf in combinations(list(range(L)), N)
            )

_cached_amplitudes = dict()

def precompute_amplitudes(L: int, N: int) -> np.array(float):
    global _cached_amplitudes
    if (L, N) in _cached_amplitudes:
        return _cached_amplitudes[(L, N)]
    _cached_amplitudes[(L, N)] = np.array([
        np.abs(amplitude(L, 1/2, np.array(conf)))
        for conf in combinations(list(range(L)), N)
    ])
    return _cached_amplitudes[(L, N)]


def reset_amp_cache():
    global _cached_amplitudes
    _cached_amplitudes = dict()


def imps_normalization_cached(L: int, N: int, alpha: float) -> float:
    return np.sum(precompute_amplitudes(L, N) ** (4 * alpha))


def imps_minents(sizes, alpha, Sz=1/2):
    minents = []
    nsizes = len(sizes)
    for i in range(nsizes):
        L = sizes[i]
        N = int((L - 2*Sz)/2)
        print(f"> imps minent for L = {L}, N = {N}, alpha = {alpha}")
        Z = imps_normalization(L, N, alpha)
        neel_state = np.array([ 2*n + 1 for n in range(N) ])
        prob = np.abs(amplitude(L, alpha, neel_state))**2 / Z
        minents.append(-np.log(prob))
    return minents


def minent_vs_alpha(deltas, sizes, Sz=1/2):
    data = pd.DataFrame(columns=("delta", "L", "minent"))
    alphas = np.arccos(-deltas) / (2 * pi)
    for alpha, delta in zip(alphas, deltas):
        print(f">>> Computing for alpha = {alpha} (delta = {delta})")
        minents = imps_minents(sizes, alpha, Sz=Sz)
        for L, minent in zip(sizes, minents):
            data.loc[len(data)] = [delta, L, minent]
    return data


def imps_norm_ratio(N, alpha):
    print(f">> N = {N} and alpha = {alpha:.4f}")
    print("    > Computing Z odd \t", end="")
    norm_odd = imps_normalization_cached(2*N + 1, N, alpha)
    print(f": {norm_odd:.4E}")
    print("    > Computing Z even\t", end="")
    norm_even = imps_normalization_cached(2*N, N, alpha)
    print(f": {norm_even:.4E}")
    ratio = norm_odd / norm_even
    print(f"    > Ratio at N = {N} \t: {ratio:.6f}")
    return ratio, norm_odd, norm_even


def norm_ratio_scaling(N_range, deltas):
    data = pd.DataFrame(columns=("delta", "N", "ratio", "norm_odd", "norm_even"))
    deltas = np.array(deltas)
    alphas = np.arccos(-deltas) / (2 * pi)
    for alpha, delta in zip(alphas, deltas):
        print(f">>> Computing norm ratio for alpha = {alpha} (delta = {delta})")
        for N in N_range:
            ratio, odd, even = imps_norm_ratio(N, alpha)
            data.loc[len(data)] = [delta, N, ratio, odd, even]
    return data
