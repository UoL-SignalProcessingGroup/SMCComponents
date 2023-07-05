import numpy as np

from smccomponents.utils import log_sum_exp


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

    if comm:
        log_likelihood = log_sum_exp(logw[index])
    else:
        log_likelihood = np.max(logw[index]) + np.log(np.sum(np.exp(logw[index] - np.max(logw[index]))))

    # Normalise the weights
    wn = np.exp(logw[index] - log_likelihood)

    return wn, log_likelihood  # type: ignore
