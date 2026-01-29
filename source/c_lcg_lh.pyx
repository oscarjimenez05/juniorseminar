# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import math
import sys
from libc.string cimport memmove

from libc.stdlib cimport malloc, free

import cython

np.import_array()
cdef unsigned long long[21] FACTORIALS
FACTORIALS[0] = 1;
FACTORIALS[1] = 1;
FACTORIALS[2] = 2;
FACTORIALS[3] = 6
FACTORIALS[4] = 24;
FACTORIALS[5] = 120;
FACTORIALS[6] = 720;
FACTORIALS[7] = 5040
FACTORIALS[8] = 40320;
FACTORIALS[9] = 362880;
FACTORIALS[10] = 3628800
FACTORIALS[11] = 39916800;
FACTORIALS[12] = 479001600;
FACTORIALS[13] = 6227020800
FACTORIALS[14] = 87178291200;
FACTORIALS[15] = 1307674368000
FACTORIALS[16] = 20922789888000;
FACTORIALS[17] = 355687428096000
FACTORIALS[18] = 6402373705728000;
FACTORIALS[19] = 121645100408832000
FACTORIALS[20] = 2432902008176640000

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline tuple __g_lcg_lh64_internal(
        unsigned long long seed, int n, long long minimum,
        long long maximum, int w, int delta, int debug):
    """
    Internal C-level function. Returns (results_array, next_seed).
    """

    cdef long long r = maximum-minimum+1

    if delta == 0:
        delta = w

    cdef unsigned long long lcg_a = 6364136223846793005
    cdef unsigned long long lcg_c = 1442695040888963407
    cdef unsigned long long x = seed

    cdef unsigned long long R = FACTORIALS[w]
    cdef unsigned long long thresh = R - (R % r)
    cdef np.ndarray[np.int64_t, ndim=1] lehmer_codes = np.empty(shape=n, dtype=np.int64)
    cdef long long * lehmer_codes_ptr = <long long *> lehmer_codes.data

    cdef unsigned long long lehmer
    cdef int i, j, k, smaller
    cdef int count = 0

    if w > 20: raise ValueError("w must be <= 20 for optimized version")

    cdef unsigned long long underl_sequence[20]
    cdef long long previous_digits[20]
    cdef long long current_digits[20]

    cdef long long * p_prev = previous_digits
    cdef long long * p_curr = current_digits
    cdef long long * p_temp
    cdef long long new_digit


    for i in range(w):
        x = lcg_a * x + lcg_c
        underl_sequence[i] = x

    cdef bint is_initialized = 0

    while count < n:
        # lehmer from scratch
        if delta == w or (not is_initialized):
            lehmer = 0
            is_initialized = 1
            for i in range(w):
                smaller = 0
                for j in range(i + 1, w):
                    smaller += (underl_sequence[j] < underl_sequence[i])
                if i < w - 1:
                    p_prev[i] = smaller
                lehmer += <unsigned long long> smaller * FACTORIALS[w - 1 - i]
        else:
            lehmer = 0

            # update the surviving digits
            for i in range(w - delta - 1):
                new_digit = p_prev[i + delta]
                for k in range(w - delta, w):
                    new_digit += (underl_sequence[k] < underl_sequence[i])
                p_curr[i] = new_digit
                lehmer += <unsigned long long> new_digit * FACTORIALS[w - 1 - i]

            for i in range(w - delta - 1, w):
                smaller = 0
                for j in range(i + 1, w):
                    smaller += (underl_sequence[j] < underl_sequence[i])
                p_curr[i] = smaller
                lehmer += <unsigned long long> smaller * FACTORIALS[w - 1 - i]

            p_temp = p_prev;
            p_prev = p_curr;
            p_curr = p_temp

        if lehmer < thresh:
            lehmer_codes_ptr[count] = (lehmer % r) + minimum
            count += 1

        if debug:
            print(f"Base sequence: {underl_sequence}")
            print(f"Next seed: {seed}")
            print(f"Lehmer code: {lehmer} (valid? {lehmer < thresh})")
            print(f"Lehmer code adjusted for range: {(lehmer % r) + minimum})")
            py_prev_digits = [previous_digits[i] for i in range(w - 1)]
            print(f"Previous digits: {py_prev_digits}")
            print("\n----------\n")

        if delta < w:
            memmove(underl_sequence, underl_sequence + delta, (w - delta) * 8)

        for i in range(w - delta, w):
            x = lcg_a * x + lcg_c
            underl_sequence[i] = x

    return (lehmer_codes, x)

cpdef tuple g_lcg_lh64(unsigned long long seed, int n, long long minimum,
                       long long maximum, int w, int delta, int debug):
    """
    Wrapper returns (numpy_array, next_seed)
    """
    return __g_lcg_lh64_internal(seed, n, minimum, maximum, w, delta, debug)


cpdef int calculate_w(long long r):
    cdef int w = 1
    cdef unsigned long long f = 1
    while True:
        w += 1
        f *= w
        if f >= r and (f % r) <= (0.05 * f):
            break
    return w

