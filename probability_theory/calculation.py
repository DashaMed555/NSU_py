import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("Дарья Ильинична Медведева.txt") as f:
        data = [float(num) for num in f.readlines()]

    print("If we consider the use of Kolmogorov criterion, then")
    if Kolmogorov(data):
        print("this sample most likely belongs to a uniform distribution on the interval [0; 1]")
    else:
        print("this sample most likely does not belong to a uniform distribution on the interval [0; 1]")

    print("\nIf we consider the use of Pearson criterion, then")
    if Pearson(data):
        print("this sample most likely belongs to a uniform distribution on the interval [0; 1]")
    else:
        print("this sample most likely does not belong to a uniform distribution on the interval [0; 1]")


def Kolmogorov(data):
    n = len(data)

    def F_empirical(t):
        indicators = [elem < t for elem in data]
        return sum(indicators) / n

    def K(t, lim=3):
        return sum(((-1) ** j * np.exp(-2 * j ** 2 * t ** 2) for j in range(-lim, lim + 1)))

    sqrt_n = math.sqrt(n)
    absolute_difference_in_jump_locations = [[abs(F_empirical(elem) - F_o(elem)),
                                              abs(F_empirical(elem + 1E-16) - F_o(elem))]
                                             for elem in data]
    unpacked = sum(absolute_difference_in_jump_locations, [])
    sup = max(unpacked)
    k = sqrt_n * sup
    print("\tK = ", k)
    p_value = 1 - K(k)
    print("\tp_value = ", p_value)
    e = 0.1
    a = 1.23
    print(f"\tFor e = {e}: a = {a}")
    e = 0.01
    a = 1.63
    print(f"\tFor e = {e}: a = {a}")
    e = 0.001
    a = 1.95
    print(f"\tFor e = {e}: a = {a}")
    # plt.plot(sorted(data), [F_empirical(elem) for elem in sorted(data)], 'ro')
    # plt.plot(sorted(data), [F_o(elem) for elem in sorted(data)])
    # plt.show()


def Pearson(data, k=12):
    n = len(data)
    intervals = [(i/k, (i+1)/k) for i in range(k)]
    frequencies = [sum([intervals[i][0] <= elem < intervals[i][1] for elem in data]) for i in range(k - 1)]
    frequencies.append(sum([intervals[k - 1][0] <= elem <= intervals[k - 1][1] for elem in data]))
    probabilities = [F_o(intervals[i][1]) - F_o(intervals[i][0]) for i in range(k)]
    psi_n = sum([((frequencies[i] - n * probabilities[i])**2)/(n * probabilities[i]) for i in range(k)])
    print(f"\tpsi_n = {psi_n}")
    p_value = 0
    print("\tp_value = ", p_value)
    e = 0.1
    a = 17.275
    print(f"\tFor e = {e}: a = {a}")
    e = 0.01
    a = 24.725
    print(f"\tFor e = {e}: a = {a}")
    e = 0.001
    a = 31.264
    print(f"\tFor e = {e}: a = {a}")


def F_o(t):
    if t < 0:
        return 0
    if t > 1:
        return 1
    return t


main()
