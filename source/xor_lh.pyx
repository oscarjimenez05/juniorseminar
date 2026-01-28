# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import math
import sys
from libc.string cimport memmove
from libc.stdlib cimport malloc, free

import cython
from libc.stdint cimport uint64_t

np.import_array()

cdef inline uint64_t xorshift64_next(uint64_t *state) nogil:
    cdef uint64_t x = state[0]
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    state[0] = x
    return x
# ------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline np.ndarray[np.int64_t, ndim=1] __xor_lh_internal(
        unsigned long long seed, int n, long long minimum,
        long long maximum, int w, int delta, int debug):
    """
    Internal C-level function with inlined Xorshift64.
    """

    cdef long long r = maximum - minimum + 1

    if seed == 0: seed = 123456789
    cdef uint64_t state = seed

    if delta == 0:
        delta = w

    cdef unsigned long long *factorials = get_factorials(w)
    cdef unsigned long long R = math.factorial(w)
    cdef unsigned long long thresh = R - (R % r)
    cdef np.ndarray[np.int64_t, ndim=1] lehmer_codes = np.empty(shape=n, dtype=np.int64)
    cdef np.int64_t * lehmer_codes_ptr = &lehmer_codes[0]

    cdef unsigned long long lehmer
    cdef int i, j, k, smaller
    # counting numbers in final array
    cdef int count = 0

    if delta > w:
        print(f"Delta {delta} greater than window size {w}", sys.stderr)
        exit(1)

    cdef np.ndarray[np.uint64_t, ndim=1] underl_sequence = np.empty(w, dtype=np.uint64)
    cdef unsigned long long * underl_ptr = &underl_sequence[0]

    for i in range(w):
        underl_ptr[i] = xorshift64_next(&state)
    # ---------------------------------

    # for incremental updates
    cdef bint is_initialized = 0
    cdef long long * previous_digits = <long long *> malloc(w * sizeof(long long))
    cdef long long * current_digits = <long long *> malloc(w * sizeof(long long))
    if not previous_digits or not current_digits:
        raise MemoryError()
    cdef long long * temp_ptr
    cdef long long new_digit

    while count < n:
        if delta == w or (not is_initialized):
            lehmer = 0
            is_initialized = 1
            for i in range(w):
                smaller = 0
                for j in range(i + 1, w):
                    if underl_ptr[j] < underl_ptr[i]:
                        smaller += 1

                # storing digits for recomputation
                if i < w - 1:
                    previous_digits[i] = smaller
                lehmer += <unsigned long long> smaller * factorials[i]

        # update lehmer, not from scratch
        else:
            lehmer = 0

            # update the surviving digits
            for i in range(w - delta - 1):
                new_digit = previous_digits[i + delta]

                # O(w) pass for each of the new ones
                for k in range(w - delta, w):
                    if underl_ptr[k] < underl_ptr[i]:
                        new_digit += 1

                current_digits[i] = new_digit
                lehmer += <unsigned long long> new_digit * factorials[i]

            # delta new digits from scratch
            for i in range(w - delta - 1, w):
                smaller = 0
                for j in range(i + 1, w):
                    if underl_ptr[j] < underl_ptr[i]:
                        smaller += 1

                current_digits[i] = smaller
                lehmer += <unsigned long long> smaller * factorials[i]

            # swap buffers
            temp_ptr = previous_digits
            previous_digits = current_digits
            current_digits = temp_ptr

        ########## check for bounds
        if lehmer < thresh:
            lehmer_codes_ptr[count] = (lehmer % r) + minimum
            count += 1

        if debug:
            print(f"Base sequence: {underl_sequence}")
            print(f"Next seed: {state}")
            print(f"Lehmer code: {lehmer} (valid? {lehmer<thresh})")
            print(f"Lehmer code adjusted for range: {(lehmer%r) + minimum})")
            py_prev_digits = [previous_digits[i] for i in range(w - 1)]
            print(f"Previous digits: {py_prev_digits}")
            print("\n----------\n")

        if delta < w:
            memmove(underl_ptr,
                    underl_ptr + delta,
                    (w - delta) * sizeof(unsigned long long))

        # generate new numbers into the pointer
        for i in range(w - delta, w):
            underl_ptr[i] = xorshift64_next(&state)

    free(previous_digits)
    free(current_digits)
    free(factorials)
    return lehmer_codes

cpdef np.ndarray[np.int64_t, ndim=1] xor_lh(unsigned long long seed, int n, long long minimum,
                                            long long maximum, int w, int delta,
                                            int debug):
    """
    Wrapper for Xorshift64 Lehmer Generator.
    """
    return __xor_lh_internal(seed, n, minimum, maximum, w, delta, debug)


cdef unsigned long long* get_factorials(int w):
    cdef unsigned long long * factorials = <unsigned long long *> malloc(w * sizeof(unsigned long long))
    if not factorials:
        raise MemoryError()
    cdef int i
    for i in range(w):
        factorials[i] = math.factorial(w - i - 1)
    return factorials


cdef _calculate_w (long long r, int debug=0, float alpha=0.05):
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
