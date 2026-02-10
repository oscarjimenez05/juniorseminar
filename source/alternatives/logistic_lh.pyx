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
    cdef double weyl_state
    cdef double weyl_constant

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

        # Weyl Sequence, minimal fix for stuck states
        # golden ratio fractional part: (sqrt(5)-1)/2
        self.weyl_constant = 0.618033988749895
        self.weyl_state = self.state

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

        # Pin variables
        cdef double p_state = self.state
        cdef double p_weyl = self.weyl_state
        cdef double p_const = self.weyl_constant
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

        if not self.is_initialized:
            for i in range(self.w):
                # Logistic Step
                p_state = 4.0 * p_state * (1.0 - p_state)
                # weyl Update
                p_weyl += p_const
                if p_weyl >= 1.0: p_weyl -= 1.0

                p_state += (p_weyl * 1e-12)
                if p_state >= 1.0: p_state -= 1.0

                p_window[i] = p_state
            self.is_initialized = 1

        while count < n:
            if p_delta < p_w:
                memmove(p_window, p_window + p_delta, (p_w - p_delta) * sizeof(double))

            for k in range(p_w - p_delta, p_w):
                # Logistic Step
                p_state = 4.0 * p_state * (1.0 - p_state)

                # Weyl update
                p_weyl += p_const
                if p_weyl >= 1.0: p_weyl -= 1.0

                # Perturb
                p_state += (p_weyl * 1e-12)
                if p_state >= 1.0: p_state -= 1.0

                p_window[k] = p_state

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
                print(f"Base sequence: {current_window}")
                print(f"State: {p_state}")
                print(f"Lehmer digits: {debug_digits}")
                print(f"Lehmer code: {lehmer}")
                print("\n----------\n")

        self.state = p_state
        self.weyl_state = p_weyl
        return results