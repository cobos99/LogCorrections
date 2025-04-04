from numpy import log, sin, pi


def f(j, l):
    """Main function that enters the sums"""
    return log(2 * sin(pi * j / (l + 1/2)))

def full_sum(l):
    """Full sum for the Slater determinant, without normalization constant"""
    return sum((l -j + 1) * f(j, l) for j in range(1, l+1))

def parity_delta(l):
    """Extra term that appears in `simpl_sum` that depends on the parity of `l`"""
    if l % 2 == 1:
        return ((l+1)/2) * f((l+1)//2, l)
    else:
        return 0

def simpl_sum(l):
    """Sum only up to l/2"""
    return (l + 1) * sum(f(j, l) for j in range(1, l//2 + 1)) + parity_delta(l)

def phi(l):
    """Sum of the `f(j,l)` terms without the prefactor `(l - j + 1)`"""
    return sum(f(j, l) for j in range(1, l//2 + 1))

def delta_fn(j, l):
    """Function that enter the deviation term"""
    eps = pi / (2*l + 1)
    angle = lambda j: 2 * pi * j / (2*l + 1)
    return j * log(sin(angle(j) - eps) / sin(angle(j)))

def delta(l):
    """Deviation term that exist for the odd case"""
    return sum(delta_fn(j, l) for j in range(1, l//2 + 1))

def log_norm(l):
    """Log of the normalization constant for the Slater determinant"""
    return (l + 1) * log(2 * l + 1)
