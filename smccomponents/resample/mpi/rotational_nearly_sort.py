import numpy as np
from math import log2
from mpi4py import MPI

from .rotational_split import rotate, accept


def sequential_nearly_sort(x, ncopies):
    x_new = np.zeros_like(x)
    ncopies_new = np.zeros_like(ncopies)

    mask = ncopies > 0
    pos = np.sum(mask)
    ncopies_new[0:pos] = ncopies[mask]
    x_new[0:pos] = x[mask]

    return x_new, ncopies_new, len(ncopies) - pos


def divide_and_rotate(x, ncopies, shifts, and_bit):  # and_bit in Algorithm 2 of [1] is called n2^{k-1}
    loc_n = len(ncopies)
    r = shifts & and_bit  # internal rotations to perform
    lsb = r > 0

    starter_to_send = shifts - (shifts & and_bit)
    starter_to_keep = shifts - (shifts & and_bit)  # (~lsb) * shifts

    ncopies_to_send = np.roll((np.arange(loc_n) < r) * ncopies, -r)
    ncopies_to_keep = np.roll((np.arange(loc_n) >= r) * ncopies, -r)

    x_to_send = np.roll(x * np.atleast_2d((np.arange(loc_n) < r)).transpose(), -r, axis=0)
    x_to_keep = np.roll(x * np.atleast_2d((np.arange(loc_n) >= r)).transpose(), -r, axis=0)

    return ncopies_to_send, x_to_send, starter_to_send, ncopies_to_keep, x_to_keep, starter_to_keep


def divide(ncopies, x, shifts, and_bit):  # and_bit in Algorithm 2 of [1] is called n2^{k-1}
    lsb = (shifts >> int(log2(and_bit))) & 1  # if lsb = 1 we send everything, otherwise we keep everything

    starter_to_send = shifts - (shifts & and_bit)
    starter_to_keep = int(not(bool(lsb))) * shifts  # (~lsb) * shifts

    ncopies_to_send = lsb * ncopies
    ncopies_to_keep = int(not(bool(lsb))) * ncopies  # (~lsb) * ncopies

    x_to_send = x * np.atleast_2d((ncopies_to_send > 0)).transpose()
    x_to_keep = x * np.atleast_2d((ncopies_to_keep > 0)).transpose()

    return ncopies_to_send, x_to_send, starter_to_send, ncopies_to_keep, x_to_keep, starter_to_keep


def rot_nearly_sort(x, ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    P = comm.Get_size()
    N = loc_n * P
    rank = comm.Get_rank()
    base = rank * loc_n  # In the paper is named n*p

    x, ncopies, zeros = sequential_nearly_sort(x, ncopies)

    shifts = np.zeros_like(zeros)
    shifts_MPI_dtpe = MPI._typedict[zeros.dtype.char]
    comm.Exscan(sendbuf=[zeros, shifts_MPI_dtpe], recvbuf=[shifts, shifts_MPI_dtpe], op=MPI.SUM)

    # Compute the MSB to check (top) and the LSB to check (down)
    down = max(loc_n, 1)
    max_shifts = np.zeros_like(shifts)
    comm.Allreduce(sendbuf=[shifts if rank == P-1 else np.array(0), shifts_MPI_dtpe], recvbuf=[max_shifts, shifts_MPI_dtpe], op=MPI.SUM)
    top = 1 if max_shifts == 0 else 1 << int(log2(max_shifts))  # to fix math domain error on for loop, when top is 0

    if loc_n > 1:
        dist = 1
        send_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        lsb = shifts & (down - 1) > 0

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide_and_rotate(x, ncopies, shifts, down-1)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner,
                                                    recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, np.repeat(1, loc_n))
        starter = starter_recv if lsb == 1 and recv_partner > rank else starter
        shifts = starter

    # Iterate from the LSB to the MSB
    for k in 2 ** np.array(range(int(log2(down)), int(log2(top)) + 1)):
        dist = int(k / loc_n)
        send_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank + dist) & (P - 1)
        lsb = (shifts >> int(log2(k))) & 1

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide(ncopies, x, shifts, k)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner,
                                                    recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, np.repeat(lsb == 0, loc_n))
        starter = starter_recv if lsb == 1 and recv_partner > rank else starter
        shifts = starter

    return x, ncopies
