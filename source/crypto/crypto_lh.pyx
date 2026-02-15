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

cdef inline uint64_t xorshift64_step(uint64_t x) nogil:
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x

cdef inline uint64_t rotate_left(uint64_t number, int r) nogil:
    return (number << (r & 63)) | (number >> (64 - (r & 63)))

cdef inline uint64_t mix_arx(uint64_t *states) nogil:
    cdef uint64_t s1 = states[1]
    cdef uint64_t s2 = states[2]
    cdef uint64_t s3 = states[3]
    cdef uint64_t s4 = states[4]

    s1 = s1 + s2
    s4 = s4 ^ s1
    s4 = rotate_left(s4, 24)  # tested from rotation_amount

    s3 = s3 + s4
    s2 = s2 ^ s3
    s2 = rotate_left(s2, 12)

    s1 = s1 + s2
    s4 = s4 ^ s1
    s4 = rotate_left(s4, 8)

    s3 = s3 + s4
    s2 = s2 ^ s3
    s2 = rotate_left(s2, 7)

    return s1 ^ s2 ^ s3 ^ s4

cdef class CryptoLehmer:
    cdef uint64_t *states
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

    def __cinit__(self, uint64_t[::1] states, int w, int delta, long long minimum, long long maximum):
        # Check for zero-state in the seed
        if states[1] == 0: states[1] = 123456789

        self.states = &states[0]

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)
        cdef int count = 0
        cdef int i, j, k, smaller
        cdef uint64_t lehmer
        cdef uint64_t candidate_mix
        cdef uint64_t clock_control
        cdef int digits[32]

        if not self.is_initialized:
            for i in range(self.w):
                for j in range(5):
                    self.states[j] = xorshift64_step(self.states[j])
                self.window_buffer[i] = mix_arx(self.states)
            self.is_initialized = 1

        # PINNED LOCAL VARIABLES
        cdef uint64_t *p_states = self.states
        cdef uint64_t p_thresh = self.thresh
        cdef uint64_t  p_minimum = self.minimum
        cdef uint64_t  p_r = self.r
        cdef int p_w = self.w
        cdef int p_delta = self.delta
        cdef uint64_t *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        while count < n:
            if p_delta < p_w:
                memmove(p_window,
                        p_window + p_delta,
                        (p_w - p_delta) * sizeof(uint64_t))

            k = p_w - p_delta
            while k < p_w: # must use while loop so continue doesnt update k automatically
                for j in range(5):
                    p_states[j] = xorshift64_step(p_states[j])

                # If the top 2 bits are both 0 (prob 0.25), we discard this round.
                clock_control = p_states[0] >> 62
                if clock_control == 0:
                    continue  # discard

                p_window[k] = mix_arx(p_states)
                k += 1

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
                for i in range(5):
                    print(f"State[{i}]: {p_states[i]}")
                print(f"Lehmer digits: {[digits[k] for k in range(p_w)]}")
                print(f"Lehmer code: {lehmer} (valid? {lehmer < p_thresh})")
                print(f"Lehmer code adjusted for range: {(lehmer % p_r) + p_minimum})")
                print("\n----------\n")

        # CRUCIAL, update persistent state
        self.states = p_states

        return results