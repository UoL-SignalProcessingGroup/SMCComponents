import numpy as np
from math import log2
from mpi4py import MPI
from .prefix_sum import inclusive_prefix_sum


def divide(ncopies, x, csum, starter, and_bit):  # and_bit in Algorithm 4 of [1] is called N2^{-k}
    loc_n = len(ncopies)
    base = loc_n * MPI.COMM_WORLD.Get_rank()  # base memory address for each rank (in paper it's named 'np')
    j = np.array(range(loc_n))  # array of all indexes j = 0, 1, 2, ..., loc_n-1

    do_shifts = (ncopies > 0) * (csum - base - j > and_bit)  # boolean array telling which elements to send
    ncopies_to_send = do_shifts * np.minimum(ncopies, abs(csum - base - j - and_bit))
    ncopies_to_keep = ncopies - ncopies_to_send

    x_to_send = x * np.atleast_2d((ncopies_to_send > 0)).transpose()
    x_to_keep = x * np.atleast_2d((ncopies_to_keep > 0)).transpose()

    do_shifts_idx = np.flatnonzero(do_shifts) #mask = do_shifts == False
    pivot = 0 if len(do_shifts_idx) == 0 else do_shifts_idx[0]  # Index of the first element to send

    starter_to_send = (csum[pivot] - ncopies_to_send[pivot]) * (ncopies_to_send[pivot] > 0)
    starter_to_keep = 0 if ((starter_to_send == starter) and (ncopies_to_send[pivot] > 0)) else starter

    return ncopies_to_send, x_to_send, starter_to_send, ncopies_to_keep, x_to_keep, starter_to_keep


def rotate(ncopies, x, starter, send_partner, recv_partner):
    comm = MPI.COMM_WORLD
    temp_ncopies = np.zeros_like(ncopies)
    temp_x = np.zeros_like(x)
    temp_starter = np.zeros_like(starter)
    ncopies_MPI_dtype = MPI._typedict[ncopies.dtype.char]
    x_MPI_dtype = MPI._typedict[x.dtype.char]
    starter_MPI_dtype = MPI._typedict[starter.dtype.char]

    comm.Sendrecv(sendbuf=[ncopies, ncopies_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_ncopies, ncopies_MPI_dtype], source=recv_partner, recvtag=0)
    comm.Sendrecv(sendbuf=[x, x_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_x, x_MPI_dtype], source=recv_partner, recvtag=0)
    comm.Sendrecv(sendbuf=[starter, starter_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_starter, starter_MPI_dtype], source=recv_partner, recvtag=0)

    return temp_ncopies, temp_x, temp_starter


def accept(x, ncopies, x_recv, ncopies_recv, mask):
    return x*np.atleast_2d(mask).transpose()+x_recv, mask*ncopies+ncopies_recv


def divide_and_rotate(x, ncopies, csum):
    loc_n = len(ncopies)
    rank = MPI.COMM_WORLD.Get_rank()
    base = loc_n * rank
    j = np.array(range(loc_n))

    ncopies_to_send = np.zeros_like(ncopies)
    x_to_send = np.zeros_like(x)
    x_to_keep = x

    ncopies_to_split_and_send = ((csum >= (rank + 1)*loc_n)*np.minimum(ncopies, csum - base - loc_n))
    ncopies_to_split_and_rotate = ((csum > ncopies + j + base) * (ncopies - ncopies_to_split_and_send))
    ncopies_to_keep = ncopies - (ncopies_to_split_and_send + ncopies_to_split_and_rotate)

    new_indexes = ((csum - base - ncopies_to_split_and_send) & (loc_n - 1))[np.nonzero(ncopies_to_split_and_send)]
    ncopies_to_send[new_indexes] = ncopies_to_split_and_send[np.nonzero(ncopies_to_split_and_send)]
    x_to_send[new_indexes, :] = x[np.nonzero(ncopies_to_split_and_send), :]

    new_indexes = (csum - ncopies - base)[np.nonzero(ncopies_to_split_and_rotate)]
    ncopies_to_keep[new_indexes] = ncopies_to_split_and_rotate[np.nonzero(ncopies_to_split_and_rotate)]
    x_to_keep[new_indexes, :] = x[np.nonzero(ncopies_to_split_and_rotate), :]

    return ncopies_to_send, x_to_send, ncopies_to_keep, x_to_keep


def divide_and_rotate2(x, ncopies, csum):
    loc_n = len(ncopies)
    rank = MPI.COMM_WORLD.Get_rank()
    base = loc_n * rank

    ncopies_to_send = np.zeros_like(ncopies)
    ncopies_to_keep = np.copy(ncopies)
    x_to_send = np.zeros_like(x)
    x_to_keep = np.copy(x)

    for i in range(loc_n-1, -1, -1):
        if ncopies_to_keep[i] > 0:
            new_temp_ncopies = min(ncopies_to_keep[i], csum[i] - (rank+1)*loc_n) if csum[i] >= (rank+1)*loc_n else 0
            new_ncopies = ncopies_to_keep[i] - new_temp_ncopies if csum[i] > ncopies_to_keep[i] + (i + base) else 0

            if new_temp_ncopies > 0:
                new_index = (csum[i] - base - new_temp_ncopies) & (loc_n - 1)
                ncopies_to_send[new_index] = new_temp_ncopies
                x_to_send[new_index, :] = x_to_keep[i, :]

            if new_ncopies > 0:
                new_index = i + csum[i] - ncopies_to_keep[i] - (base+i)
                ncopies_to_keep[new_index] = new_ncopies
                x_to_keep[new_index, :] = x_to_keep[i, :]
            ncopies_to_keep[i] -= (new_temp_ncopies + new_ncopies)

    return ncopies_to_send, x_to_send, ncopies_to_keep, x_to_keep


def rot_split(x, ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    P = comm.Get_size()
    N = loc_n*P
    rank = comm.Get_rank()
    base = rank*loc_n  # In the paper is named n*p

    # Initialise the cumulative sum of ncopies, and trunk to zero those elements i for which ncopies[i] = 0
    csum = inclusive_prefix_sum(ncopies)*(ncopies > 0)
    starter = csum[0] - ncopies[0]

    # Compute the MSB to check (top) and the LSB to check (down)
    down = max(loc_n, 1)  # loc_n if loc_n > 1 else 1
    max_bit = np.max((ncopies > 0)*(csum - np.array(range(loc_n)) - base))
    top = np.zeros_like(max_bit)
    max_bit_MPI_dtype = MPI._typedict[max_bit.dtype.char]
    comm.Allreduce(sendbuf=[max_bit, max_bit_MPI_dtype], recvbuf=[top, max_bit_MPI_dtype], op=MPI.MAX)
    top = top >> 1 if (top & (~top + 1)) == top else 1 << int(log2(top))  # top & (~top + 1) is the MSB
    top = max(top, 1)  # to fix math domain error on the next line, when the previous returns 0

    # Iterate from the MSB to the LSB
    for k in 2**np.array(range(int(log2(top)), int(log2(down))-1, -1)):
        # Compute the MPI ranks to send to and receive from
        dist = int(k/loc_n)  # distance (in MPI ranks) to/from which send/receive
        send_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide(ncopies, x, csum, starter, k)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner, recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, ncopies > 0)

        starter = starter_recv if (starter_recv > 0 and rank > recv_partner) else starter
        csum = starter + np.cumsum(ncopies)

    # Leaf stage of the binary tree
    if loc_n > 1:
        dist = 1
        send_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2

        ncopies_to_send, x_to_send, ncopies, x = divide_and_rotate(x, ncopies, csum)

        ncopies_recv, x_recv, _ = rotate(ncopies_to_send, x_to_send, np.array([0]), send_partner, recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, ncopies > 0)

    return x, ncopies
