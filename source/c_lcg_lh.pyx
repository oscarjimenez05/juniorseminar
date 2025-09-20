# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import math

from libc.stdlib cimport malloc, free

import cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int64_t, ndim=1] lcg(long long seed, int n,
                                          long long a=1664525, long long c=1013904223, long long m=2**32):
    """
    Cython implementation of a Linear Congruential Generator.
    :param seed: (unsigned int): initial seed
    :param n: (int): number of elements to generate
    :param a: (unsigned int): multiplier
    :param c: (unsigned int): increment
    :param m: (long long): modulus
    """
    cdef np.ndarray[np.int64_t, ndim=1] result = np.empty(n, dtype=np.int64)

    cdef long long x = seed
    cdef int i
    for i in range(n):
        x = (a * x + c) % m
        result[i] = x
    return result

def _rel_ord(np.ndarray[np.int64_t, ndim=1] sequence, int w):
    """
    Cython wrapper for generating relative orderings.
    """
    windows = np.lib.stride_tricks.sliding_window_view(sequence, w)
    ranks = np.argsort(np.argsort(windows, axis=1), axis=1)
    return ranks


@cython.boundscheck(False)
@cython.wraparound(False)
def _lehmer_from_ranks(np.ndarray[np.int64_t, ndim=2] rank_lists):
    """
    Calculates Lehmer codes from rank orderings with C loops.
    """
    cdef int num_windows = rank_lists.shape[0]
    cdef int w = rank_lists.shape[1]

    # Create the result array
    cdef np.ndarray[np.int64_t, ndim=1] results = np.empty(num_windows, dtype=np.int64)

    # Pre-calculate factorials in a C array for fast lookup
    cdef long long* factorials = <long long*>malloc(w * sizeof(long long))
    if not factorials:
        raise MemoryError()

    cdef int i, j, k
    for i in range(w):
        factorials[i] = math.factorial(w - i - 1)

    # Use a typed memoryview for fast, direct access to the rank_lists data
    cdef long long[:, ::1] ranks_view = rank_lists

    cdef long long code
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

cpdef lcg_lh(long long seed, int n, int w, long long a=1664525, long long c=1013904223, long long m=2**32):
    """
    Generates Lehmer codes from a non-overlapping sliding window over an LCG sequence.
    All intermediate steps are handled efficiently within Cython.
    """
    cdef np.ndarray[np.int64_t, ndim=1] base_sequence = lcg(seed, n + w - 1, a, c, m)

    cdef np.ndarray[np.int64_t, ndim=2] ranks = _rel_ord(base_sequence, w)

    cdef np.ndarray[np.int64_t, ndim=1] lehmer_codes = _lehmer_from_ranks(ranks)

    return lehmer_codes

