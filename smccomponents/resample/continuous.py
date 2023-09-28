import warnings
import autograd.numpy as np
from mpi4py import MPI

from .mpi.redistribution import fixed_size_redistribution
from .mpi.prefix_sum import inclusive_prefix_sum

"""
Parallel systematic resampling from [1].

[1] An O(log2N) Fully-Balanced Resampling Algorithm for Particle Filters
    on Distributed Memory Architectures (2021) Varsi, A., Maskell, S., & Spirakis, P.
    https://www.mdpi.com/1999-4893/14/12/342
"""


def get_ncopies(wn, N, comm, rng):
    """
    Determine the number of copies of each particle.

    Args:
        wn: A list of normalised weights
        comm: The MPI communicator

    Returns:
        ncopies: The number of copies of each particle
    """

    # Calculate the cdf of wn
    cdf = inclusive_prefix_sum(wn * N)

    u = rng.uniform(0, 1)
    u = comm.bcast(u, root=0)

    cdf_of_i_minus_one = cdf - (wn * N)
    ncopies = np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)
    ncopies = ncopies.astype(int)

    return ncopies


def check_stability(ncopies, N, N_local, comm):
    # Calculate the sum of ncopies across all ranks and ensure it equals N
    ncopies_sum = comm.allreduce(np.sum(ncopies), op=MPI.SUM)

    rank = comm.Get_rank()

    sum_of_ncopies = np.array(1, dtype=ncopies.dtype)
    comm.Allreduce(sendbuf=[np.sum(ncopies), MPI.INT], recvbuf=[sum_of_ncopies, MPI.INT], op=MPI.SUM)

    if sum_of_ncopies != N:
        # Find the index of the last particle to be copied
        idx = np.where(ncopies > 0)
        idx = idx[0][-1] + rank * N_local if len(idx[0]) > 0 else np.array([-1])
        idx_MPI_dtype = MPI._typedict[idx.dtype.char]
        max_idx = np.zeros(1, dtype=idx.dtype)
        comm.Allreduce(sendbuf=[idx, idx_MPI_dtype], recvbuf=[max_idx, idx_MPI_dtype], op=MPI.MAX)

        # Find the core which has that particle, and increase/decrease its ncopies[i] till sum_of_ncopies == N
        if rank * N_local <= max_idx <= (rank + 1) * N_local - 1:
            ncopies[max_idx - rank * N_local] -= sum_of_ncopies - N

    sum_of_ncopies = np.array(1, dtype=ncopies.dtype)
    comm.Allreduce(sendbuf=[np.sum(ncopies), MPI.INT], recvbuf=[sum_of_ncopies, MPI.INT], op=MPI.SUM)
    if sum_of_ncopies != N:
        raise ValueError(f"[Rank {rank}] Sum of ncopies ({ncopies_sum}) does not equal N ({N})")

    return ncopies


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
