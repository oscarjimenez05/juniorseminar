import random
import math
import secrets

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from c_lcg_lh import lcg_lh64


def lcg(seed: int, n: int, a=1664525, c=1013904223, m=2 ** 32) -> [int]:
    """
    Default range is 0,2^32-1
    param seed (int): initial seed
    param n (int): number of elements to generate
    """
    result = np.empty(n)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        result[i] = x
    return result


def xorshift(seed, n) -> [int]:
    """
    Default range is 0,2^32-1
    param seed (int): initial seed
    param n (int): number of elements to generate
    """
    result = []
    x = seed
    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        result.append(x & 0xFFFFFFFF)
    return result


def _lehmer_from_ranks(rank_lists: [[int]]) -> [int]:
    n = len(rank_lists[0])
    factorials = [math.factorial(n - i - 1) for i in range(n)]
    results = []
    for ranks in rank_lists:
        n = len(ranks)
        code = 0
        for i in range(n):
            # below calculates normal lehmer code
            smaller = sum(1 for j in range(i + 1, n) if ranks[j] < ranks[i])
            code += smaller * factorials[i]
        results.append(code)
    return results


def lcg_lh(seed: int, n: int, w: int, a=1664525, c=1013904223, m=2 ** 32) -> [int]:
    """
    Overlapping lehmer code sliding window on top of LCG
    """
    return _lehmer_from_ranks(sliding_window_view(lcg(seed, n + w - 1, a, c, m), w))


def csprng(reps: int, max_exclusive: int):
    a_csprng = np.empty(reps)
    for i in range(reps):
        a_csprng[i] = secrets.randbelow(max_exclusive)
    return a_csprng


def mrs_tw(seed: int, reps: int, max_exclusive: int):
    random.seed(seed)
    a_mrs_tw = np.empty(reps)
    for i in range(reps):
        a_mrs_tw[i] = random.randint(0, max_exclusive - 1)
    return a_mrs_tw


def pcg64(seed: int, reps: int, max_exclusive: int):
    np.random.seed(seed)
    return np.random.randint(0, max_exclusive, size=reps)


if __name__ == '__main__':
    print(lcg_lh64(42, 10, 4, 2))
