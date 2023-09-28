import numpy as np
from mpi4py import MPI


def inclusive_prefix_sum(array):
    comm = MPI.COMM_WORLD

    csum = np.cumsum(array).astype(array.dtype)
    offset = np.zeros(1, dtype=array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]
    comm.Exscan(sendbuf=[csum[-1], MPI_dtype], recvbuf=[offset, MPI_dtype], op=MPI.SUM)

    return csum + offset
