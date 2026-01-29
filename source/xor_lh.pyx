# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import math
from libc.string cimport memmove
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t

np.import_array()

cdef inline uint64_t xorshift64_step(uint64_t x) nogil:
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x

cdef class XorLehmer:
    cdef uint64_t state
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
        if seed == 0: seed = 123456789
        self.state = seed
        self.w = w

        # fully non-overlapping
        if delta == 0:
            self.delta = w
        else:
            self.delta = delta

        self.is_initialized = 0

        self.minimum = minimum
        self.maximum = maximum
        self.r = self.maximum - self.minimum + 1
        self.R = math.factorial(self.w)
        self.thresh = self.R - (self.R % self.r)

        self.window_buffer = <uint64_t *> malloc(w * sizeof(uint64_t))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0
        cdef int i, j, k, smaller
        cdef uint64_t lehmer

        cdef int digits[32]

        if not self.is_initialized:
            for i in range(self.w):
                self.state = xorshift64_step(self.state)
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        while count < n:
            if self.delta < self.w:
                memmove(self.window_buffer,
                        self.window_buffer + self.delta,
                        (self.w - self.delta) * sizeof(uint64_t))

            # generate delta new numbers using Xorshift
            for k in range(self.w - self.delta, self.w):
                self.state = xorshift64_step(self.state)
                self.window_buffer[k] = self.state

            lehmer = 0
            for i in range(self.w):
                smaller = 0
                for j in range(i + 1, self.w):
                    smaller += (self.window_buffer[j] < self.window_buffer[i])

                lehmer += smaller * self.factorials[i]
                digits[i] = smaller  # Capture for debug

            if lehmer < self.thresh:
                results[count] = (lehmer % self.r) + self.minimum
                count += 1

            if debug:
                current_window = [self.window_buffer[k] for k in range(self.w)]
                print(f"Base sequence: {current_window}")
                print(f"State: {self.state}")
                print(f"Lehmer digits: {[digits[k] for k in range(self.w)]}")
                print(f"Lehmer code: {lehmer} (valid? {lehmer < self.thresh})")
                print(f"Lehmer code adjusted for range: {(lehmer % self.r) + self.minimum})")
                print("\n----------\n")

        return results