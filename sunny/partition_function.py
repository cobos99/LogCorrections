import numpy as np
from numpy import pi

NUM_TERMS=1000

def partition_function(
    length: float,
    circumf: float,
    comp_radius: float,
    winding: str = "integer",
        velocity: float = 1.):
    ratio = velocity * circumf / length
    R = comp_radius
    if winding == "half":
        Sz_range = np.arange(-NUM_TERMS, +NUM_TERMS) + 0.5
    else:
        Sz_range = np.arange(-NUM_TERMS, +NUM_TERMS)

    m_range = np.arange(1, NUM_TERMS)

    sum_part = np.sum( np.exp( - (pi * ratio) * (2 * pi * R**2) * (Sz_range**2)) )
    mult_part = np.prod( 1 - np.exp( - pi * ratio * m_range))

    return sum_part / mult_part



def free_energy(
    length: float,
    circumf: float,
    comp_radius: float,
    winding: str = "integer",
        velocity: float = 1.):
    return -np.log(partition_function(length, circumf, comp_radius, winding, velocity))


def free_energy_diff(
    length: float,
    circumf: float,
    comp_radius: float,
    ):
    return free_energy(length, circumf, comp_radius, winding="integer") - free_energy(length, circumf, comp_radius, winding="half")


def entropy(
    length: float,
    circumf: float,
    comp_radius: float,
    winding: str = "integer",
    ):
    return - 2 * free_energy(length, circumf, comp_radius, winding) \
        + free_energy(2 * length, circumf, comp_radius, winding)


def entropy_diff(
    length: float,
    circumf: float,
    comp_radius: float,
    ):
    return entropy(length, circumf, comp_radius, winding="integer") - entropy(length, circumf + 1, comp_radius, winding="half")
