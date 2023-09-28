import numpy as np
from mpi4py import MPI
from .rotational_nearly_sort import rot_nearly_sort
from .rotational_split import rot_split
from smccomponents.utils import pad, restore


def sequential_redistribution(x, ncopies):
    return np.repeat(x, ncopies, axis=0)


def fixed_size_redistribution(x, ncopies):

    if MPI.COMM_WORLD.Get_size() > 1:
        x, ncopies = rot_nearly_sort(x, ncopies)
        x, ncopies = rot_split(x, ncopies)

    x = sequential_redistribution(x, ncopies)

    return x


def variable_size_redistribution(particles, ncopies):
    x = pad(particles)

    x = fixed_size_redistribution(x, ncopies)

    particles = restore(x, particles)

    return particles
