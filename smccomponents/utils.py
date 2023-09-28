import numpy as np

try:
    from mpi4py import MPI
    mpi_avail = True
except ImportError:
    mpi_avail = False

from scipy.special import logsumexp

from .tree import Tree


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


def normalise_weights(logw, comm=None):
    """
    Normalises the sample weights

    Args:
        logw: A list of sample weights on the log scale
        comm: MPI communicator

    Returns:
        A list of normalised weights

    """

    index = ~np.isneginf(logw)

    # Calculate the log likelihood
    if comm:
        log_likelihood = log_sum_exp(logw[index])
    else:
        log_likelihood = logsumexp(logw[index])

    # Normalise the weights
    wn = np.zeros_like(logw)
    wn[index] = np.exp(logw[index] - log_likelihood)

    return wn, log_likelihood  # type: ignore
