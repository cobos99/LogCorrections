from numpy import log, sin, pi


def f(j, l):
    """Main function that enters the sums"""
    return log(2 * sin(pi * j / l))

def full_sum(l):
    """Full sum for the Slater determinant, without normalization constant"""
    return sum( (l - j) * f(j, l) for j in range(1, l))

def parity_delta(l):
    """Extra term that appears in `simpl_sum` that depends on the parity of `l`"""
    if l % 2 == 0:
        return f(l//2, l)
    else:
        return 0

def simpl_sum(l):
    """Simplified sum for the Slater determinant"""
    return l * sum( f(j, l) for j in range(1, l//2 + 1)) - (l/2) * parity_delta(l)

def phi(l):
    """Sum of the `f(j,l)` terms without the prefactor `(l - j + 1)`"""
    return sum(f(j, l) for j in range(1, l//2 + 1))

def log_norm(l):
    """Log of the normalization constant for the Slater determinant"""
    return l * log(2 * l)
