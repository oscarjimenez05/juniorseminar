# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import math
from libc.string cimport memmove
from libc.stdlib cimport malloc, free

np.import_array()

cdef class LcgLehmer:
    cdef unsigned long long state
    cdef unsigned long long a
    cdef unsigned long long c
    cdef unsigned long long *window_buffer
    cdef unsigned long long *factorials
    cdef int w
    cdef int delta
    cdef bint is_initialized

    cdef long long minimum
    cdef long long maximum
    cdef long long r
    cdef unsigned long long R
    cdef unsigned long long thresh

    def __cinit__(self, unsigned long long seed, int w, int delta, long long minimum, long long maximum):
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

        self.window_buffer = <unsigned long long *> malloc(w * sizeof(unsigned long long))
        self.factorials = <unsigned long long *> malloc(w * sizeof(unsigned long long))

        # precompute factorials
        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)

    cpdef np.ndarray generate_chunk(self, int n, int debug):

        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0
        cdef int i, j, smaller
        cdef unsigned long long lehmer

        if not self.is_initialized:
            for i in range(self.w):
                self.state = self.a * self.state + self.c
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        while count < n:
            # shift window left by delta elements (unless fully replacing it)
            if self.delta < self.w:
                memmove(self.window_buffer,
                        self.window_buffer + self.delta,
                        (self.w - self.delta) * sizeof(unsigned long long))

            # generate delta new numbers at the end
            for k in range(self.w - self.delta, self.w):
                self.state = self.a * self.state + self.c
                self.window_buffer[k] = self.state

            # calculate Lehmer Code
            lehmer = 0
            for i in range(self.w):
                smaller = 0
                for j in range(i + 1, self.w):
                    smaller += (self.window_buffer[j] < self.window_buffer[i])
                lehmer += smaller * self.factorials[i]

            if lehmer < self.thresh:
                results[count] = (lehmer % self.r) + self.minimum
                count += 1

            if debug:
                current_window = [self.window_buffer[k] for k in range(self.w)]
                print(f"Base sequence: {current_window}")
                print(f"State: {self.state}")
                print(f"Lehmer code: {lehmer} (valid? {lehmer < self.thresh})")
                print(f"Lehmer code adjusted for range: {(lehmer % self.r) + self.minimum})")
                print("\n----------\n")

        return results