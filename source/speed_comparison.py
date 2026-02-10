from collections import Counter
import time
from generators import *
from stat_properties import display_arrays
import matplotlib.pyplot as plt

import c_lcg_lh, xor_lh
from source.alternatives import lcg_fenwick, xor_fenwick


def shannon_entropy(seq: [int]):
    n = len(seq)
    counts = Counter(seq)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return entropy


def speed_test(disp=False):
    reps = 1_000_000
    window_range = 14
    seed = 123456789

    max_exclusive = 2**32

    # CSPRNG
    start_csprng = time.perf_counter()
    a_csprng = csprng(reps, max_exclusive)
    end_csprng = time.perf_counter()
    assert len(a_csprng) == reps

    # LCG_LH
    LCG_LH = c_lcg_lh.LcgLehmer(seed, window_range, 0, 0, max_exclusive-1)
    start_lcg_lh = time.perf_counter()
    # fully non-overlapping
    a_lcg_lh = LCG_LH.generate_chunk(reps, 0)
    end_lcg_lh = time.perf_counter()
    a_lcg_lh = np.array(a_lcg_lh)
    assert len(a_lcg_lh) == reps

    # XOR_LH
    XOR_LH = xor_lh.XorLehmer(seed, window_range, 0, 0, max_exclusive - 1)
    start_xor_lh = time.perf_counter()
    # fully non-overlapping
    a_xor_lh = XOR_LH.generate_chunk(reps, 0)
    end_xor_lh = time.perf_counter()
    a_xor_lh = np.array(a_xor_lh)
    assert len(a_xor_lh) == reps

    # LCG_FENWICK
    LCG_FW = lcg_fenwick.LcgFenwick(seed, window_range, 0, 0, max_exclusive - 1)
    start_lcg_fw = time.perf_counter()
    a_lcg_fw = LCG_FW.generate_chunk(reps, 0)
    end_lcg_fw = time.perf_counter()
    a_lcg_fw = np.array(a_lcg_fw)
    assert len(a_lcg_fw) == reps

    # XOR_FENWICK
    XOR_FW = xor_fenwick.XorFenwick(seed, window_range, 0, 0, max_exclusive - 1)
    start_xor_fw = time.perf_counter()
    a_xor_fw = XOR_FW.generate_chunk(reps, 0)
    end_xor_fw = time.perf_counter()
    a_xor_fw = np.array(a_xor_fw)
    assert len(a_xor_fw) == reps

    # MRS_TW
    start_mrs_tw = time.perf_counter()
    a_mrs_tw = mrs_tw(seed, reps, max_exclusive)
    end_mrs_tw = time.perf_counter()
    assert len(a_mrs_tw) == reps

    # PCG64
    start_pcg64 = time.perf_counter()
    a_pcg64 = pcg64(seed, reps, max_exclusive)
    end_pcg64 = time.perf_counter()
    assert len(a_pcg64) == reps

    print("Average time for CSPRNG: " + str((end_csprng - start_csprng) / reps))
    print("Average time for LCG_LH: " + str((end_lcg_lh - start_lcg_lh) / reps))
    print("Average time for XOR_LH: " + str((end_xor_lh - start_xor_lh) / reps))
    print("Average time for LCG_FW: " + str((end_lcg_fw - start_lcg_fw) / reps))
    print("Average time for XOR_FW: " + str((end_xor_fw - start_xor_fw) / reps))
    print("Average time for MRW_TW: " + str((end_mrs_tw - start_mrs_tw) / reps))
    print("Average time for PCG_64: " + str((end_pcg64 - start_pcg64) / reps))

    if disp:
        display_arrays([("CSPRNG", a_csprng),
                        ("LCG_LH", a_lcg_lh),
                        ("XOR_LH", a_xor_lh),
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
        a_lcg_lh = c_lcg_lh.g_lcg_lh64(seed, reps, 0, math.factorial(window_range)-1, window_range, i, 0)
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


def calc_alpha_star():
    reps = 1_000_000
    seed = 123456789
    times = []
    for i in range(2, 19):
        start = time.perf_counter()
        a_lcg_lh = c_lcg_lh.g_lcg_lh64(seed, reps, 0, math.factorial(i) // 2, i, 0, 0)
        end = time.perf_counter()
        a_lcg_lh = np.array(a_lcg_lh)
        assert len(a_lcg_lh) == reps
        times.append(end - start)

    print("w\tL(w)\t\tL(w+1)\t\ta*")
    for i in range(len(times) - 1):
        L_w = (i + 2) * (i + 1) / 2
        L_w1 = (i + 3) * (i + 2) / 2
        alpha_star = (times[i + 1] - times[i]) / (L_w1 - L_w)
        print(f"{i + 2}\t{times[i]:.6f}\t{times[i + 1]:.6f}\t{alpha_star:.6f}")

    x = np.arange(len(times))
    slope, intercept = np.polyfit(x, times, 1)
    print(f"Slope: {slope}")
    plt.plot(times)
    plt.ylim(bottom=0)
    plt.show()


def compare_window_sizes():
    reps = 1_000_000
    seed = 123456789
    print("w,r,R,TotalTime,alpha")
    for i in range(2, 19):
        r = 2 * math.factorial(i) // 3
        for j in range(2, 19):
            if math.factorial(j) >= r:
                start = time.perf_counter()
                a_lcg_lh = c_lcg_lh.g_lcg_lh64(seed, reps, 1, r, j, 0, 0)
                end = time.perf_counter()
                a_lcg_lh = np.array(a_lcg_lh)
                assert len(a_lcg_lh) == reps
                avg_time = (end - start)
                print(f"{i},{r},{j},{avg_time:.6f},{(math.factorial(j)%r)/math.factorial(j):.6f}")


if __name__ == "__main__":
    speed_test()
    # compare_cython_speed()
    # compare_overlap_speed()
    # calc_alpha_star()
    # compare_window_sizes()
