import c_lcg_lh
from stat_properties import general_display_arrays


def max_case_check():
    reps = 10
    seed = 2**63

    a_lcg_lh = c_lcg_lh.lcg_lh64(seed, reps, 18, 18)
    assert len(a_lcg_lh) == reps

    if reps <= 100:
        for i in range(1, reps):
            if a_lcg_lh[i] < a_lcg_lh[i-1]:
                print("\n\nIt does loop\n\n")
                break
        print(a_lcg_lh)


def g_lcg_lh64_check():
    seed = 123456789
    reps = 10000
    minimum = -728
    maximum = 4999
    a_g_lcg_lh64 = c_lcg_lh.g_lcg_lh64(seed, reps, minimum, maximum, 1)
    general_display_arrays([("Generalized LCG_LH", a_g_lcg_lh64)], minimum, maximum)

if __name__ == '__main__':
    # max_case_check()
    g_lcg_lh64_check()
