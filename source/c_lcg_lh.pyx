import numpy as np
cimport numpy as np

np.import_array()

cpdef np.ndarray[np.uint32_t, ndim=1] lcg(unsigned int seed, int n,
                                          unsigned int a=1664525, unsigned int c=1013904223, long long m=2**32):
    """
    Cython implementation of a Linear Congruential Generator.
    :param seed: (unsigned int): initial seed
    :param n: (int): number of elements to generate
    :param a: (unsigned int): multiplier
    :param c: (unsigned int): increment
    :param m: (long long): modulus
    """
    cdef np.ndarray[np.uint32_t, ndim=1] result = np.empty(n, dtype=np.uint32)

    cdef unsigned int x = seed
    cdef int i

    for i in range(n):
        x = (a * x + c) % m
        result[i] = x

    return result
