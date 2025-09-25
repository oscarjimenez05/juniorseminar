from typing import Tuple
import matplotlib.pyplot as plt
from generators import *
import c_lcg_lh
from scipy.stats import chisquare
import statsmodels.api as sm


def large_lcg_vs_lcg_lh():
    seed = 701
    reps = 100000
    window = 6

    max_exclusive = math.factorial(window)

    a_lcg = c_lcg_lh.lcg(seed, reps, a=421, c=1, m=max_exclusive)
    a_large_lcg = c_lcg_lh.lcg(seed, reps)
    a_lcg_mod = np.array(list(map(lambda x: x % max_exclusive, a_large_lcg)))
    a_lcg_lh = np.array(c_lcg_lh.lcg_lh(seed, reps, window))
    a_csprng = csprng(reps, max_exclusive)
    display_arrays([("CSPRNG", a_csprng), ("LCG   ", a_lcg), ("Lm_LCG", a_lcg_mod), ("LCG_LH", a_lcg_lh)], max_exclusive)

    # Plot histograms
    plot_distribution(a_csprng, "CSPRNG", 2*9*5)
    plot_distribution(a_lcg, "LCG", 2*9*5)
    plot_distribution(a_lcg_mod, "Lm_LCG", 2*9*5)
    plot_distribution(a_lcg_lh, "LCG_LH", 2*9*5)

    # chisq test
    print("-----------------------")
    counts, _ = np.histogram(a_csprng, bins=max_exclusive, range=(0, max_exclusive))
    chi2, p = chisquare(counts)
    print(f"CSPRNG Chi^2 statistic = {chi2:.2f}, p-value = {p:.5f}")
    counts, _ = np.histogram(a_lcg, bins=max_exclusive, range=(0, max_exclusive))
    chi2, p = chisquare(counts)
    print(f"LCG    Chi^2 statistic = {chi2:.2f}, p-value = {p:.5f}")
    counts, _ = np.histogram(a_lcg_mod, bins=max_exclusive, range=(0, max_exclusive))
    chi2, p = chisquare(counts)
    print(f"Lm_LCG Chi^2 statistic = {chi2:.2f}, p-value = {p:.5f}")
    counts, _ = np.histogram(a_lcg_lh, bins=max_exclusive, range=(0, max_exclusive))
    chi2, p = chisquare(counts)
    print(f"LCG_LH Chi^2 statistic = {chi2:.2f}, p-value = {p:.5f}")

    # serial correlation test
    # print("-----------------------")
    # ljung_box_results = sm.stats.acorr_ljungbox(a_lcg, lags=[1, window, max_exclusive], return_df=True)
    # print(ljung_box_results)
    # ljung_box_results = sm.stats.acorr_ljungbox(a_lcg_mod, lags=[1, window, max_exclusive], return_df=True)
    # print(ljung_box_results)
    # ljung_box_results = sm.stats.acorr_ljungbox(a_lcg_lh, lags=[1, window, max_exclusive], return_df=True)
    # print(ljung_box_results)


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
            plt.xlabel("Index of value generated")
            plt.ylabel("Value Generated")
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


def plot_distribution(data, title="Distribution of Values", bins=24):
    plt.hist(data, bins=bins, align="left", rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xlabel("Generated Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    large_lcg_vs_lcg_lh()
