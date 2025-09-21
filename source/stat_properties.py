from typing import Tuple
import matplotlib.pyplot as plt
from generators import *
import c_lcg_lh


def large_lcg_vs_lcg_lh():
    seed = 12345789
    reps = 10000000
    window = 9

    max_exclusive = math.factorial(window)

    a_large_lcg = c_lcg_lh.lcg(seed, reps)
    a_lcg_mod = np.array(list(map(lambda x: x % max_exclusive, a_large_lcg)))
    a_lcg_lh = np.array(c_lcg_lh.lcg_lh(seed, reps, window))
    display_arrays([("Lm_LCG", a_lcg_mod), ("LCG_LH", a_lcg_lh)], max_exclusive)


def missing_from_range(lst: [int], start: int, end: int) -> [int]:
    """
    :param lst: a list of numbers
    :param start: start value (inclusive)
    :param end: end value (inclusive)
    :return: a list of numbers not present in the range
    """
    full_range = set(range(start, end + 1))
    return sorted(full_range - set(lst))


def display_arrays(data: [Tuple[str, list]], max_exclusive: int, plot: bool = False) -> None:
    """
    :param data: a list of tuples of the form (title, array)
    :param max_exclusive: range of numbers generated, 0 to max_exclusive-1
    :param plot: whether to plot the data or not, defaults to False
    :return: None
    """
    if plot:
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
    exp_mean = (max_exclusive - 1) / 2
    print("Expec. MEAN: " + str(exp_mean))
    for title, array in data:
        mean = array.mean()
        print(f"{title} MEAN: " + str(mean) + f" (diff. {(exp_mean - mean):.5f})")


def plot_distribution(data, title="Distribution of values", bins=24):
    plt.hist(data, bins=bins, range=(0, 24), align="left", rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xlabel("Lehmer code values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(24))
    plt.show()


if __name__ == "__main__":
    large_lcg_vs_lcg_lh()
