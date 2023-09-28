import warnings
import autograd.numpy as np
from mpi4py import MPI

from .mpi.redistribution import fixed_size_redistribution
from .mpi.ncopies import get_ncopies, check_stability


def resample(x, wn, N, N_local, comm, log_likelihood=None, rng=np.random.default_rng()):
    """
    Resample the particles.

    Args:
        x: A list of samples to resample
        logw: A set of log weights
        N: The total number of samples
        N_local: The number of samples on this rank
        comm: The MPI communicator
        log_likelihood: The log likelihood of the current samples
        rng: A random number generator

    Returns:
        x_new: A list of resampled samples
        logw_new: Weights after resampling
    """

    # Resample x
    ncopies = get_ncopies(wn, N, comm, rng)

    # Check stability of ncopies
    ncopies = check_stability(ncopies, N, N_local, comm)

    # Redistribute the samples
    x_new = fixed_size_redistribution(x, ncopies)

    if len(x_new) != N_local:
        raise ValueError(f"Number of resampled samples ({len(x_new)}) does not equal N_local ({N_local})")

    # Determine new weights
    if log_likelihood:
        logw_new = (np.ones(N_local) * log_likelihood) - np.log(N_local)
    else:
        logw_new = np.log(np.ones(N_local)) - np.log(N_local)

    return x_new, logw_new
