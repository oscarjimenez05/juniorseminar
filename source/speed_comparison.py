from collections import Counter
import time
from generators import *
from stat_properties import display_arrays
import matplotlib.pyplot as plt

import c_lcg_lh


def shannon_entropy(seq: [int]):
    n = len(seq)
    counts = Counter(seq)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return entropy


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
    a_lcg = c_lcg_lh.lcg(seed, reps, a=421, c=1, m=max_exclusive)
    end_lcg = time.perf_counter()
    assert len(a_lcg) == reps

    # LCG_LH
    start_lcg_lh = time.perf_counter()
    a_lcg_lh = c_lcg_lh.lcg_lh64(seed, reps, window_range, window_range) # fully non-overlapping
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
                   max_exclusive, True)


def compare_cython_speed():
    reps = 100000
    start_c_old = time.perf_counter()
    old = c_lcg_lh.lcg_lh(135, reps, 6)
    end_c_old = time.perf_counter()
    assert len(old) == reps

    start_c_new = time.perf_counter()
    new = c_lcg_lh.lcg_lh64(135, reps, 6)
    end_c_new = time.perf_counter()
    assert len(new) == reps

    start_p = time.perf_counter()
    lcg_lh(135, reps, 5, 121, 1, 720)
    end_p = time.perf_counter()

    time_c_new = end_c_new - start_c_new
    time_c_old = end_c_old - start_c_old
    time_p = end_p - start_p
    print(time_c_new)
    print(time_c_old)
    print(time_p)
    print(time_p/time_c_new)
    print(time_c_old/time_c_new)


def compare_overlap_speed():
    reps = 100000
    window_range = 10
    seed = 123456789

    times = []
    for i in range(1, window_range+1):
        start = time.perf_counter()
        a_lcg_lh = c_lcg_lh.lcg_lh64(seed, reps, window_range, i)
        end = time.perf_counter()
        a_lcg_lh = np.array(a_lcg_lh)
        assert len(a_lcg_lh) == reps
        times.append(end-start)

    x = np.arange(len(times))
    slope, intercept = np.polyfit(x, times, 1)
    print(f"Slope: {slope}")
    plt.plot(times)
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    speed_test()
    # compare_cython_speed()
    # compare_overlap_speed()
