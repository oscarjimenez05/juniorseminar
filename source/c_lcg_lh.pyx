# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
from fenwick import FenwickTree
import math
import sys
import fenwick

from libc.stdlib cimport malloc, free

import cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint64_t, ndim=1] lcg(unsigned long long seed, int n,
                                          unsigned long long a=1664525,
                                          unsigned long long c=1013904223,
                                          unsigned long long m=4294967296):
    """
    Cython implementation of a Linear Congruential Generator.
    :param seed: (unsigned int): initial seed
    :param n: (int): number of elements to generate
    :param a: (unsigned int): multiplier
    :param c: (unsigned int): increment
    :param m: (long long): modulus
    """
    cdef np.ndarray[np.uint64_t, ndim=1] result = np.empty(n, dtype=np.uint64)

    cdef unsigned long long x = seed
    cdef int i
    for i in range(n):
        x = (a * x + c) % m
        result[i] = x
    return result


cpdef np.ndarray[np.uint64_t, ndim=1] lcg64(unsigned long long seed, int n):
    """
    Cython implementation of a Linear Congruential Generator mod 2^64.
    It uses parameters a and c from Knuth's MMIX LCG
    :param seed: (unsigned lonng long int): initial seed
    :param n: (int): number of elements to generate
    """
    cdef np.ndarray[np.uint64_t, ndim=1] result = np.empty(n, dtype=np.uint64)
    cdef unsigned long long a = 6364136223846793005
    cdef unsigned long long c = 1442695040888963407
    cdef unsigned long long x = seed
    cdef int i
    for i in range(n):
        x = ( a * x + c) & 0xFFFFFFFFFFFFFFFF
        result[i] = x
    return result

cdef np.ndarray[np.uint64_t, ndim=2]_rel_ord(np.ndarray[np.uint64_t, ndim=1] sequence, int w):
    """
    Cython wrapper for generating relative orderings.
    """
    windows = np.lib.stride_tricks.sliding_window_view(sequence, w)
    ranks = np.argsort(np.argsort(windows, axis=1), axis=1)
    return ranks.astype(np.uint64)


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef np.ndarray[np.uint64_t, ndim=1] _lehmer_from_ranks(np.ndarray[np.uint64_t, ndim=2] rank_lists):
    """
    Calculates Lehmer codes from rank orderings with C loops.
    """
    cdef int num_windows = rank_lists.shape[0]
    cdef int w = rank_lists.shape[1]

    # Create the result array
    cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(num_windows, dtype=np.uint64)

    # Pre-calculate factorials in a C array for fast lookup
    cdef unsigned long long* factorials = <unsigned long long*>malloc(w * sizeof(unsigned long long))
    if not factorials:
        raise MemoryError()

    cdef int i, j, k
    for i in range(w):
        factorials[i] = math.factorial(w - i - 1)

    # Use a typed memoryview for fast, direct access to the rank_lists data
    cdef const np.uint64_t[:, :] ranks_view = rank_lists

    cdef unsigned long long code
    cdef int smaller


    for i in range(num_windows):
        code = 0
        for j in range(w):
            smaller = 0

            for k in range(j + 1, w):
                if ranks_view[i, k] < ranks_view[i, j]:
                    smaller += 1
            code += smaller * factorials[j]
        results[i] = code

    free(factorials)

    return results

cpdef lcg_lh(unsigned long long seed, int n, int w, unsigned long long a=1664525,
             unsigned long long c=1013904223, unsigned long long m=4294967296):
    """
    Generates Lehmer codes from a non-overlapping sliding window over an LCG sequence.
    All intermediate steps are handled efficiently within Cython.
    Default underlying LCG is standard [0,2^32-1]
    """
    cdef np.ndarray[np.uint64_t, ndim=1] base_sequence = lcg(seed, n + w - 1, a, c, m)

    cdef np.ndarray[np.uint64_t, ndim=2] ranks = _rel_ord(base_sequence, w)
    cdef np.ndarray[np.uint64_t, ndim=2] ranks_uint64 = ranks.astype(np.uint64)

    cdef np.ndarray[np.uint64_t, ndim=1] lehmer_codes = _lehmer_from_ranks(ranks_uint64)

    return lehmer_codes

cpdef np.ndarray[np.uint64_t, ndim=1] lcg_lh64(unsigned long long seed, int n, int w, int delta=1,
                                               int debug = 0):
    """
    Generates Lehmer codes from a sliding window over an LCG sequence with customizable delta.
    Underlying LCG range of 2^64.
    """
    if delta>w:
        print(f"Delta {delta} greater than window size {w}", sys.stderr)

    cdef np.ndarray[np.uint64_t, ndim=1] base_sequence = lcg64(seed, (n - 1) * delta + w)

    cdef np.ndarray[np.uint64_t, ndim=2] windows = np.lib.stride_tricks.sliding_window_view(
        base_sequence, w, axis=0)[::delta]

    if debug:
        print(f"Base sequence: {base_sequence}")
        print(f"Generated windows with deltas of {delta}: {windows}")

    if windows.shape[0] > n:
        windows = windows[:n]

    cdef np.ndarray[np.uint64_t, ndim=1] lehmer_codes = _lehmer_from_ranks(windows)
    return lehmer_codes

cpdef np.ndarray[np.uint64_t, ndim=1] calc_g_lcg_lh64(long long seed, int n, long long minimum,
                                                long long maximum, int delta=1,
                                               int debug = 0):
    """
    This version calculates optimal w
    :param seed: seed
    :param n: number of numbers to generate
    :param minimum: minimum number in desired range (can be negative)
    :param maximum: maximum number in desired range (can be negative)
    :param delta: sliding window overlap
    :param debug: whether to print debug statements
    :return: the array of generated numbers
    """

    cdef int w = _calculate_w(maximum-minimum+1, debug=debug)
    return g_lcg_lh64(seed, n, minimum, maximum, w, delta, debug)

cpdef np.ndarray[np.uint64_t, ndim=1] g_lcg_lh64(long long seed, int n, long long minimum,
                                                long long maximum, int w, int delta,
                                               int debug):
    """
    General version which takes all parameters
    LCG_LH implementation for generalized ranges.
    Underlying LCG range of 2^64.
    It generates only w LCG numbers at a time.
    """

    cdef long long r = maximum-minimum+1

    if delta == 0:
        delta = w

    cdef long long *factorials = get_factorials()
    cdef long long R = math.factorial(w)
    cdef long long thresh = R - (R%r)
    cdef np.ndarray[np.int64_t, ndim=1] lehmer_codes = np.empty(shape=n, dtype=np.int64)
    cdef long long lehmer

    if delta>w:
        print(f"Delta {delta} greater than window size {w}", sys.stderr)

    # this will count the number of final numbers in the array
    count = 0

    cdef np.ndarray[np.uint64_t, ndim=1] underl_sequence = lcg64(seed, w)

    while count < n:

        if (1):
            # lehmer update
            lehmer = 0
            pass

        else:
            # lehmer from scratch
            lehmer = 0

        if lehmer < thresh:
            lehmer_codes[count] = (lehmer%r) + minimum
            count += 1

        seed = underl_sequence[-1]

        if debug:
            print(f"Base sequence: {underl_sequence}")
            print(f"Next seed: {seed}")

        # generate the next numbers
        if delta == w:
            underl_sequence[:] = lcg64(seed, w)
        else:
            underl_sequence[:-delta] = underl_sequence[delta:]
            underl_sequence[-delta:] = lcg64(seed, delta)

    free(factorials)
    return lehmer_codes


cdef get_factorials(int w):
    cdef long long * factorials = <long long *> malloc(w * sizeof(long long))
    if not factorials:
        raise MemoryError()
    cdef int i
    for i in range(w):
        factorials[i] = math.factorial(w - i - 1)
    return factorials

cdef _calculate_w (long long r, float alpha=0.05, int debug=0):
    w = 1
    factorial = 1.0
    while 1:
        w += 1
        factorial *= w
        if debug:
            print(f"Currently on R = {factorial} ({w}!)\t\t {(factorial % r)*100/factorial} <= {100*alpha}")
        if (factorial % r <= (alpha * factorial)) and (factorial >= r):
            break
    return w

