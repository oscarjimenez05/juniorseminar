import c_lcg_lh


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


def calculate_w_check():
    print("----------")
    print(c_lcg_lh.calculate_w(361, 0.05, 1))


if __name__ == '__main__':
    max_case_check()
    calculate_w_check()