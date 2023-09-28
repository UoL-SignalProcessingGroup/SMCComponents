import numpy as np
from mpi4py import MPI
from .prefix_sum import inclusive_prefix_sum


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