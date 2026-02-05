# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import math
from libc.string cimport memmove, memset
from libc.stdlib cimport malloc, free, qsort
from libc.stdint cimport uint64_t

np.import_array()

cdef struct Element:
    uint64_t value
    int original_index

# relative ordering (compression)
cdef int compare_elements(const void *a, const void *b) noexcept nogil:
    cdef uint64_t val_a = (<Element *> a).value
    cdef uint64_t val_b = (<Element *> b).value
    if val_a < val_b: return -1
    if val_a > val_b: return 1
    return 0

cdef inline uint64_t xorshift64_step(uint64_t x) nogil:
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x

cdef class XorFenwick:
    cdef uint64_t state
    cdef uint64_t *window_buffer
    cdef uint64_t *factorials

    cdef int w
    cdef int delta
    cdef bint is_initialized

    cdef long long minimum
    cdef long long maximum
    cdef uint64_t r
    cdef uint64_t thresh

    cdef Element *sort_buffer
    cdef int *rank_buffer
    cdef int *fenwick_tree

    def __cinit__(self, uint64_t seed, int w, int delta, long long minimum, long long maximum):
        """
        :param seed: initial state
        :param w: window size
        :param delta: steps to take between windows. delta=0 is the same as delta=w
        :param minimum: inclusive
        :param maximum: inclusive
        """
        if seed == 0: seed = 123456789
        self.state = seed
        self.w = w

        if delta == 0:
            self.delta = w
        else:
            self.delta = delta

        self.minimum = minimum
        self.maximum = maximum
        self.r = <uint64_t> (self.maximum - self.minimum + 1)

        cdef uint64_t R_val = math.factorial(self.w)
        self.thresh = R_val - (R_val % self.r)

        self.window_buffer = <uint64_t *> malloc(w * sizeof(uint64_t))
        self.factorials = <uint64_t *> malloc(w * sizeof(uint64_t))

        # fenwick buffers
        self.sort_buffer = <Element *> malloc(w * sizeof(Element))
        self.rank_buffer = <int *> malloc(w * sizeof(int))
        self.fenwick_tree = <int *> malloc((w + 1) * sizeof(int))

        # precompute factorials
        cdef int i
        for i in range(w):
            self.factorials[i] = math.factorial(w - i - 1)

        self.is_initialized = 0

    def __dealloc__(self):
        if self.window_buffer: free(self.window_buffer)
        if self.factorials: free(self.factorials)
        if self.sort_buffer: free(self.sort_buffer)
        if self.rank_buffer: free(self.rank_buffer)
        if self.fenwick_tree: free(self.fenwick_tree)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray generate_chunk(self, int n, int debug):
        cdef np.ndarray[np.uint64_t, ndim=1] results = np.empty(n, dtype=np.uint64)

        cdef int i
        if not self.is_initialized:
            for i in range(self.w):
                self.state = xorshift64_step(self.state)
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        # Pin variables
        cdef uint64_t p_state = self.state
        cdef uint64_t p_thresh = self.thresh
        cdef int p_delta = self.delta
        cdef uint64_t *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        # fenwick pointers
        cdef Element *p_sort = self.sort_buffer
        cdef int *p_ranks = self.rank_buffer
        cdef int *p_bit = self.fenwick_tree

        cdef uint64_t p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef int p_w = self.w

        cdef int count = 0
        cdef int k, rank, idx, s_val
        cdef uint64_t lehmer

        while count < n:
            # shift window
            if p_delta < p_w:
                memmove(p_window, p_window + p_delta, (p_w - p_delta) * sizeof(uint64_t))

            # fill values
            for k in range(p_w - p_delta, p_w):
                p_state = xorshift64_step(p_state)
                p_window[k] = p_state

            # coordinate compression. turn numbers into ranks [1...w]
            for k in range(p_w):
                p_sort[k].value = p_window[k]
                p_sort[k].index = k

            # sort to determine relative order
            qsort(p_sort, p_w, sizeof(Element), compare_elements)

            # fill rank buffer (p_ranks[original_index] = rank) using 1-based indexing
            for k in range(p_w):
                p_ranks[p_sort[k].index] = k + 1

            # calculate
            lehmer = 0
            # reset Fenwick Tree O(W)
            memset(p_bit, 0, (p_w + 1) * sizeof(int))

            # iterate backwards from right to left
            for k in range(p_w - 1, -1, -1):
                rank = p_ranks[k]

                # Query BIT: count elements smaller than current rank seen so far (to the right)
                s_val = 0
                idx = rank - 1
                while idx > 0:
                    s_val += p_bit[idx]
                    idx -= idx & (-idx)

                lehmer += s_val * p_factorials[k]

                # add current element
                # update(rank, +1)
                idx = rank
                while idx <= p_w:
                    p_bit[idx] += 1
                    idx += idx & (-idx)

            # threshold rejection
            if lehmer < p_thresh:
                results[count] = (lehmer % p_r) + p_minimum
                count += 1

        self.state = p_state
        return results