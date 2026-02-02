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

cdef class LcgLehmer:
    cdef uint64_t state
    cdef uint64_t a
    cdef uint64_t c
    cdef uint64_t *window_buffer
    cdef uint64_t *factorials
    cdef int w
    cdef int delta
    cdef bint is_initialized

    cdef long long minimum
    cdef long long maximum
    cdef long long r
    cdef uint64_t R
    cdef uint64_t thresh

    def __cinit__(self, uint64_t seed, int w, int delta, long long minimum, long long maximum):
        """
        :param seed: initial state
        :param w: window size
        :param delta: steps to take between windows. delta=0 is the same as delta=w
        :param minimum: inclusive
        :param maximum: inclusive
        """
        self.state = seed
        self.w = w
        self.a = 6364136223846793005
        self.c = 1442695040888963407
        
        self.is_initialized = 0
        
        if delta == 0:
            self.delta = w
        else:
            self.delta = delta

        self.minimum = minimum
        self.maximum = maximum
        self.r = self.maximum - self.minimum + 1
        self.R = math.factorial(self.w)
        self.thresh = self.R - (self.R % self.r)

        self.window_buffer = <uint64_t *> malloc(w * sizeof(uint64_t))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        # precompute factorials
        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0
        cdef int i, j, k, smaller
        cdef uint64_t lehmer

        cdef int digits[32]

        if not self.is_initialized:
            for i in range(self.w):
                self.state = self.a * self.state + self.c
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        # PINNED LOCAL VARIABLES
        cdef uint64_t p_state = self.state
        cdef uint64_t p_thresh = self.thresh
        cdef uint64_t p_lehmer
        cdef long long p_minimum = self.minimum
        cdef long long p_r = self.r
        cdef int p_w = self.w
        cdef int p_delta = self.delta
        cdef uint64_t *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials
        cdef uint64_t p_a = self.a
        cdef uint64_t p_c = self.c

        while count < n:
            # shift window left by delta elements (unless fully replacing it)
            if p_delta < p_w:
                memmove(p_window,
                        p_window + p_delta,
                        (p_w - p_delta) * sizeof(uint64_t))

            # generate delta new numbers at the end
            for k in range(p_w - p_delta, p_w):
                p_state = p_a * p_state + p_c
                p_window[k] = p_state

            # calculate Lehmer Code
            lehmer = 0
            for i in range(p_w):
                smaller = 0
                for j in range(i + 1, p_w):
                    smaller += (p_window[j] < p_window[i])
                lehmer += smaller * p_factorials[i]
                digits[i] = smaller

            if lehmer < p_thresh:
                results[count] = (lehmer % p_r) + p_minimum
                count += 1

            if debug:
                current_window = [p_window[k] for k in range(p_w)]
                print(f"Base sequence: {current_window}")
                print(f"State: {p_state}")
                print(f"Lehmer digits: {[digits[k] for k in range(p_w)]}")
                print(f"Lehmer code: {lehmer} (valid? {lehmer < p_thresh})")
                print(f"Lehmer code adjusted for range: {(lehmer % p_r) + p_minimum})")
                print("\n----------\n")

        # CRUCIAL, update persistent state
        self.state = p_state

        return results