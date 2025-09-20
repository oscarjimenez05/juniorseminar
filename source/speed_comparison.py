from collections import Counter
import matplotlib.pyplot as plt
import time
from typing import Tuple
from generators import *

import c_lcg_lh


def shannon_entropy(seq: [int]):
    n = len(seq)
    counts = Counter(seq)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return entropy


def plot_distribution(data, title="Distribution of values", bins=24):
    plt.hist(data, bins=bins, range=(0, 24), align="left", rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xlabel("Lehmer code values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(24))
    plt.show()


def missing_from_range(lst: [int], start: int, end: int) -> [int]:
    """
    :param lst: a list of numbers
    :param start: start value (inclusive)
    :param end: end value (inclusive)
    :return: a list of numbers not present in the range
    """
    full_range = set(range(start, end + 1))
    return sorted(full_range - set(lst))


def display_arrays(data: [Tuple[str, list]], max_exclusive: int) -> None:
    """
    :param data: a list of tuples of the form (title, array)
    :param max_exclusive: range of numbers generated, 0 to max_exclusive-1
    :return: None
    """
    for title, array in data:
        plt.plot(array)
        plt.title(title)
        plt.show()

    print("-----------------------")
    for title, array in data:
        print(f"{title} MIN and MAX: " + str(int(array.min())) + ", " + str(int(array.max())))
    print("-----------------------")
    for title, array in data:
        print(f"Not present in {title}: " + str(missing_from_range(array, 0, max_exclusive - 1)))
    print("-----------------------")
    print("Expec. MEAN: " + str((max_exclusive - 1) / 2))
    for title, array in data:
        print(f"{title} MEAN: " + str(array.mean()))


def speed_test():
    reps = 100000
    window_range = 6
    seed = 123456789

    max_exclusive = math.factorial(window_range)

    # CSPRNG
    start_csprng = time.perf_counter()
    a_csprng = csprng(reps, max_exclusive)
    end_csprng = time.perf_counter()
    assert len(a_csprng) == reps

    # LCG
    start_lcg = time.perf_counter()
    a_lcg = c_lcg_lh.lcg(seed, reps, a=121, c=1, m=max_exclusive)
    end_lcg = time.perf_counter()
    assert len(a_lcg) == reps

    # LCG_LH
    start_lcg_lh = time.perf_counter()
    a_lcg_lh = c_lcg_lh.lcg_lh(seed, reps, window_range)
    end_lcg_lh = time.perf_counter()
    a_lcg_lh = np.array(a_lcg_lh)
    assert len(a_lcg_lh) == reps

    # MRS_TW
    start_mrs_tw = time.perf_counter()
    a_mrs_tw = mrs_tw(seed, reps, max_exclusive)
    end_mrs_tw = time.perf_counter()
    assert len(a_mrs_tw) == reps

    print("Average time for CSPRNG: " + str((end_csprng - start_csprng) / reps))
    print("Average time for LCG   : " + str((end_lcg - start_lcg) / reps))
    print("Average time for LCG_LH: " + str((end_lcg_lh - start_lcg_lh) / reps))
    print("Average time for MRW_TW: " + str((end_mrs_tw - start_mrs_tw) / reps))

    display_arrays([("CSPRNG", a_csprng),
                    ("LCG   ", a_lcg),
                    ("LCG_LH", a_lcg_lh),
                    ("MRS_TW", a_mrs_tw)],
                   max_exclusive)


def compare_cython():
    reps = 100000
    start_c = time.perf_counter()
    a_c = c_lcg_lh.lcg_lh(135, reps, 5, 121, 1, 720)
    end_c = time.perf_counter()

    start_p = time.perf_counter()
    a_p = lcg_lh(135, reps, 5, 121, 1, 720)
    end_p = time.perf_counter()

    time_c = end_c - start_c
    time_p = end_p - start_p
    print(time_c)
    print(time_p)
    print(time_p/time_c)


# rel_ord([5,4,3,1,2,9,8,12,35,15],4)
# print(lehmer_from_ranks([[2,1,0,3],[5,3,0,1,2,4]]))
# comparison()
if __name__ == "__main__":
    speed_test()
    # compare_cython()
