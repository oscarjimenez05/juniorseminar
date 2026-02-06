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
    int index

# Comparator for qsort
cdef int compare_elements(const void *a, const void *b) noexcept nogil:
    cdef uint64_t val_a = (<Element *> a).value
    cdef uint64_t val_b = (<Element *> b).value
    if val_a < val_b: return -1
    if val_a > val_b: return 1
    return 0

cdef class LcgFenwick:
    cdef uint64_t state
    cdef uint64_t a
    cdef uint64_t c
    cdef uint64_t *window_buffer
    cdef uint64_t *factorials

    cdef Element *sort_buffer
    cdef int *rank_buffer
    cdef int *fenwick_tree

    cdef int w
    cdef int delta
    cdef bint is_initialized

    cdef long long minimum
    cdef long long maximum
    cdef uint64_t r
    cdef uint64_t thresh

    def __cinit__(self, uint64_t seed, int w, int delta, long long minimum, long long maximum):
        self.state = seed
        self.w = w
        self.a = 6364136223846793005
        self.c = 1442695040888963407

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

        self.sort_buffer = <Element *> malloc(w * sizeof(Element))
        self.rank_buffer = <int *> malloc(w * sizeof(int))
        self.fenwick_tree = <int *> malloc((w + 1) * sizeof(int))

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
                self.state = self.a * self.state + self.c
                self.window_buffer[i] = self.state
            self.is_initialized = 1

        # Pin variables
        cdef uint64_t p_state = self.state
        cdef uint64_t p_a = self.a
        cdef uint64_t p_c = self.c
        cdef uint64_t p_thresh = self.thresh
        cdef int p_delta = self.delta
        cdef uint64_t *p_window = self.window_buffer
        cdef uint64_t *p_factorials = self.factorials

        cdef Element *p_sort = self.sort_buffer
        cdef int *p_ranks = self.rank_buffer
        cdef int *p_bit = self.fenwick_tree

        cdef long long p_minimum = self.minimum
        cdef uint64_t p_r = self.r
        cdef int p_w = self.w

        cdef int count = 0
        cdef int k, rank, idx, s_val
        cdef uint64_t lehmer

        while count < n:
            if p_delta < p_w:
                memmove(p_window, p_window + p_delta, (p_w - p_delta) * sizeof(uint64_t))

            for k in range(p_w - p_delta, p_w):
                p_state = p_a * p_state + p_c
                p_window[k] = p_state

            for k in range(p_w):
                p_sort[k].value = p_window[k]
                p_sort[k].index = k

            qsort(p_sort, p_w, sizeof(Element), compare_elements)

            for k in range(p_w):
                p_ranks[p_sort[k].index] = k + 1

            lehmer = 0
            memset(p_bit, 0, (p_w + 1) * sizeof(int))

            for k in range(p_w - 1, -1, -1):
                rank = p_ranks[k]

                s_val = 0
                idx = rank - 1
                while idx > 0:
                    s_val += p_bit[idx]
                    idx -= idx & (-idx)

                lehmer += s_val * p_factorials[k]

                idx = rank
                while idx <= p_w:
                    p_bit[idx] += 1
                    idx += idx & (-idx)

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
        return results