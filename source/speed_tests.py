from collections import Counter
import math
import matplotlib.pyplot as plt
from random import randint
import secrets
import time
import numpy as np


def shannon_entropy(seq: [int]):
    n = len(seq)
    counts = Counter(seq)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return entropy


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


def rel_ord(sequence: [int], w: int) -> [[int]]:
    """
    Generates all relative orderings for a sequence (not Lehmer codes)
    param w (int): window length
    """
    res = []
    for j in range(len(sequence) - w + 1):
        sub = sequence[j:(j + w)]
        sorted_idx = sorted(range(w), key=lambda i: sub[i])
        # print(sorted_idx)
        ranks = [0] * w
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank
        res.append(ranks)
    return res


def lehmer_from_ranks(rank_lists: [[int]]) -> [int]:
    lehmer_codes = []
    for ranks in rank_lists:
        n = len(ranks)
        code = 0
        for i in range(n):
            # below calculates normal lehmer code
            smaller = sum(1 for j in range(i + 1, n) if ranks[j] < ranks[i])
            # turns lehmer code into integer (lexicographic index)
            code += smaller * math.factorial(n - i - 1)
        lehmer_codes.append(code)
    return lehmer_codes


def lcg_lh(seed: int, n: int, w: int, a=1664525, c=1013904223, m=2 ** 32) -> [int]:
    """
    Non-overlapping lehmer code sliding window on top of LCG
    """
    return lehmer_from_ranks(rel_ord(lcg(seed, n + w - 1, a, c, m), w))


def plot_distribution(data, title="Distribution of values", bins=24):
    plt.hist(data, bins=bins, range=(0, 24), align="left", rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xlabel("Lehmer code values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(24))
    plt.show()


def test():
    window = 4
    a_lcg = lcg(123456, 2 ** 16)
    a_xorshift = xorshift(123456, 2 ** 16)
    a_mersenne = [randint(0, 2 ** 32 - 1) for _ in range(2 ** 16)]
    print("LCG default:\t\t" + str(shannon_entropy(a_lcg)))
    print("XOR default:\t\t" + str(shannon_entropy(a_xorshift)))
    print("Mersenne default:\t" + str(shannon_entropy(a_mersenne)))

    lehmer_lcg = lehmer_from_ranks(rel_ord(a_lcg, window))
    lehmer_xor = lehmer_from_ranks(rel_ord(a_xorshift, window))
    lehmer_mersenne = lehmer_from_ranks(rel_ord(a_mersenne, window))
    print("LCG changed:\t\t" + str(shannon_entropy(lehmer_lcg)))
    print("XOR changed:\t\t" + str(shannon_entropy(lehmer_xor)))
    print("Mersenne changed:\t" + str(shannon_entropy(lehmer_mersenne)))

    plot_distribution(lehmer_lcg, "LCG Lehmer Code Distribution (p=4)")
    plot_distribution(lehmer_xor)
    plot_distribution(lehmer_mersenne)


def speed_test():
    reps = 100000
    window_range = 6

    max_exclusive = math.factorial(window_range)

    # CSPRNG
    a_csprng = np.empty(reps)
    start_csprng = time.perf_counter()
    for i in range(reps):
        a_csprng[i] = secrets.randbelow(max_exclusive)
    end_csprng = time.perf_counter()
    assert len(a_csprng) == reps

    # LCG
    seed = 123456789
    start_lcg = time.perf_counter()
    a_lcg = lcg(seed, reps, m=max_exclusive)
    end_lcg = time.perf_counter()
    assert len(a_lcg) == reps

    # LCG_LH
    start_lcg_lh = time.perf_counter()
    a_lcg_lh = lcg_lh(seed, reps, window_range)
    end_lcg_lh = time.perf_counter()
    a_lcg_lh = np.array(a_lcg_lh)
    assert len(a_lcg_lh) == reps

    # MRS_TW
    a_mrs_tw = np.empty(reps)
    start_mrs_tw = time.perf_counter()
    for i in range(reps):
        a_mrs_tw[i] = randint(0, max_exclusive - 1)
    end_mrs_tw = time.perf_counter()
    assert len(a_mrs_tw) == reps

    print("Average time for CSPRNG: " + str((end_csprng - start_csprng) / reps))
    print("Average time for LCG   : " + str((end_lcg - start_lcg) / reps))
    print("Average time for LCG_LH: " + str((end_lcg_lh - start_lcg_lh) / reps))
    print("Average time for MRW_TW: " + str((end_mrs_tw - start_mrs_tw) / reps))
    print("-----------------------")
    print("CSPRNG MIN and MAX: " + str(int(a_csprng.min())) + ", " + str(int(a_csprng.max())))
    print("LCG    MIN and MAX: " + str(int(a_lcg.min())) + ", " + str(int(a_lcg.max())))
    print("LCH_LH MIN and MAX: " + str(int(a_lcg_lh.min())) + ", " + str(int(a_lcg_lh.max())))
    print("MRS_TW MIN and MAX: " + str(int(a_mrs_tw.min())) + ", " + str(int(a_mrs_tw.max())))
    print("-----------------------")
    print("Expec. MEAN: " + str((max_exclusive - 1) / 2))
    print("CSPRNG MEAN: " + str(a_csprng.mean()))
    print("LCG    MEAN: " + str(a_lcg.mean()))
    print("LCG_LH MEAN: " + str(a_lcg_lh.mean()))
    print("MRS_TW MEAN: " + str(a_mrs_tw.mean()))


# rel_ord([5,4,3,1,2,9,8,12,35,15],4)
# print(lehmer_from_ranks([[2,1,0,3],[5,3,0,1,2,4]]))
# test()
speed_test()
