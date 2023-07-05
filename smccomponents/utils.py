import autograd.numpy as np
from mpi4py import MPI
from scipy.special import logsumexp


def LSE(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype='d')
    y = np.frombuffer(ymem, dtype='d')
    y[:] = logsumexp(np.hstack((x, y)))


def log_sum_exp(array):
    op = MPI.Op.Create(LSE, commute=True)
    log_sum = np.zeros_like(1, array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]
    leaf_node = np.array([-np.inf]).astype(array.dtype) if len(array) == 0 else logsumexp(array)

    MPI.COMM_WORLD.Allreduce(sendbuf=[leaf_node, MPI_dtype], recvbuf=[log_sum, MPI_dtype], op=op)

    op.Free()

    return log_sum
