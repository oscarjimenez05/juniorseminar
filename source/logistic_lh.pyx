# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import math
from libc.string cimport memmove
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t

np.import_array()

cdef class LogisticLehmer:
    cdef double state
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
        """
        Logistic Map Generator using Lehmer Codes.
        State is double precision in range (0, 1).
        Map: x = 4.0 * x * (1 - x)
        """
        # normalize integer seed, multiply by 2^-64 to map full uint64 range to [0,1]
        self.state = <double> seed * 5.42101086242752217E-20

        # Safety: Logistic map gets stuck at 0 or 1.
        if self.state <= 0.0 or self.state >= 1.0:
            self.state = 0.5

        self.w = w
        self.delta = delta if delta != 0 else w

        self.minimum = minimum
        self.maximum = maximum
        self.r = <uint64_t> (self.maximum - self.minimum + 1)

        # Calculate Threshold
        cdef uint64_t R_val = math.factorial(self.w)
        self.thresh = R_val - (R_val % self.r)

        # window buffer is dOUBLE, factorials are UINT64
        self.window_buffer = <double *> malloc(w * sizeof(double))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        # Precompute factorials
        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

        self.is_initialized = 0

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)

        cdef int i
        if not self.is_initialized:
            for i in range(self.w):
                self.state = 4.0 * self.state * (1.0 - self.state)
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        # Pin variables
        cdef double p_state = self.state
        cdef uint64_t p_thresh = self.thresh
        cdef int p_delta = self.delta
        cdef double *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        cdef long long p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef int p_w = self.w

        cdef int count = 0
        cdef int j, k, smaller
        cdef uint64_t lehmer

        while count < n:
            # shift window
            if p_delta < p_w:
                memmove(p_window, p_window + p_delta, (p_w - p_delta) * sizeof(double))

            # generate new Chaotic Doubles
            for k in range(p_w - p_delta, p_w):
                p_state = 4.0 * p_state * (1.0 - p_state)
                p_window[k] = p_state

            # LEHMER
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
                print(f"Base sequence (Double): {current_window}")
                print(f"State: {p_state}")
                print(f"Lehmer digits: {debug_digits}")
                print(f"Lehmer code: {lehmer}")
                print("\n----------\n")

        self.state = p_state
        return results