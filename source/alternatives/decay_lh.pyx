# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import math
from libc.string cimport memmove
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t
from libc.math cimport log

np.import_array()

cdef class DecayLehmer:
    cdef uint64_t state
    cdef double *window_buffer
    cdef uint64_t *factorials

    cdef int w
    cdef int delta
    cdef bint is_initialized

    cdef long long minimum
    cdef long long maximum
    cdef uint64_t r
    cdef uint64_t thresh

    def __cinit__(self, uint64_t seed, int w, int delta, long long minimum, long long maximum):
        if seed == 0: seed = 123456789
        self.state = seed
        self.w = w
        self.delta = delta if delta != 0 else w

        self.minimum = minimum
        self.maximum = maximum
        self.r = <uint64_t> (self.maximum - self.minimum + 1)

        cdef uint64_t R_val = math.factorial(self.w)
        self.thresh = R_val - (R_val % self.r)

        self.window_buffer = <double *> malloc(w * sizeof(double))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

        self.is_initialized = 0

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)

        cdef uint64_t p_state = self.state
        cdef double *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        cdef long long p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef uint64_t p_thresh = self.thresh
        cdef int p_delta = self.delta
        cdef int p_w = self.w

        cdef int count = 0
        cdef int i, j, k, smaller
        cdef uint64_t lehmer
        cdef double u_val

        cdef uint64_t LCG_A = 6364136223846793005
        cdef uint64_t LCG_C = 1442695040888963407
        cdef double DBL_CONVERT = 1.0 / 9007199254740992.0

        if not self.is_initialized:
            for i in range(self.w):
                p_state = p_state * LCG_A + LCG_C
                u_val = <double> (p_state >> 11) * DBL_CONVERT
                if u_val < 1e-9: u_val = 1e-9
                # exponential decay: -ln(u)
                p_window[i] = -log(u_val)
            self.is_initialized = 1

        while count < n:
            if p_delta < p_w:
                memmove(p_window, p_window + p_delta, (p_w - p_delta) * sizeof(double))

            for k in range(p_w - p_delta, p_w):
                p_state = p_state * LCG_A + LCG_C
                u_val = <double> (p_state >> 11) * DBL_CONVERT
                if u_val < 1e-9: u_val = 1e-9
                # exponential decay
                p_window[k] = -log(u_val)

            lehmer = 0
            for i in range(p_w):
                smaller = 0
                for j in range(i + 1, p_w):
                    smaller += (p_window[j] < p_window[i])
                lehmer += smaller * p_factorials[i]

            if lehmer < p_thresh:
                results[count] = (lehmer % p_r) + p_minimum
                count += 1

            if debug:
                debug_digits = []
                for i in range(p_w):
                    s_debug = 0
                    for j in range(i + 1, p_w):
                        s_debug += (p_window[j] < p_window[i])
                    debug_digits.append(s_debug)
                current_window = [p_window[k] for k in range(p_w)]
                print(f"Decay inputs: {current_window}")
                print(f"Lehmer code: {lehmer}")
                print("\n----------\n")

        self.state = p_state
        return results