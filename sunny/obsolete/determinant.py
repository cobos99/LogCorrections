import numpy as np
import matplotlib.pyplot as plt
import approx as app
from numpy import pi

def wv(m, N: int):
    n = np.arange(N)
    return np.exp(4j*pi*n*m / (2*N - 1)) / np.sqrt(N)

def B(N: int):
    wvs = np.array([ wv(m, N) for m in range(N) ])
    return wvs.T.conj() @ wvs


marker_options = dict(markersize=8, markeredgewidth=3)
@app.fit.with_fit("log of determinant")
def log_det(Lmin, Lmax):
    sizes = np.arange(Lmin, Lmax)
    log_dets = -np.log(np.array([np.linalg.det(B(size)).real for size in sizes]))
    plt.plot(sizes, log_dets, "x", label="numerical", **marker_options)
    plt.xlabel("$N$")
    plt.ylabel("$ - \log \mathrm{det} \mathsf (B)$")
    plt.title("$S_{\infty} difference$")
    plt.legend()
    return sizes, log_dets
