import random
import numpy as np
import math
import matplotlib.pyplot as plt


def task1(lambda_, n, N):
    p = min(lambda_ / n, 1)
    success_nums = []
    distances = []
    for _ in range(N):
        bernoulli = [1 if random.random() < p else 0 for _ in range(n)]
        success_nums.append(sum(bernoulli))
        prev_success_num = n
        for i in range(n):
            if bernoulli[i] == 1:
                prev_success_num = i
                break
        for i in range(prev_success_num + 1, n):
            if bernoulli[i] == 1:
                distances.append((i - prev_success_num) / n)
                prev_success_num = i
    plt.subplot(3, 2, 1)
    m1 = 1 + math.floor(math.log2(N))
    plt.title("Task 1")
    plt.xlabel("Numbers of successes")
    plt.hist(success_nums, m1, density=True)
    plt.subplot(3, 2, 2)
    m2 = 1 + math.floor(math.log2(len(distances)))
    plt.xlabel("Distances")
    plt.hist(distances, m2, density=True)


def task2(lambda_, N):
    success_nums = []
    distances = []
    for _ in range(N):
        x = np.random.poisson(lambda_)
        successes = np.random.uniform(size=x)
        successes.sort()
        success_nums.append(x)
        distances.extend([successes[i + 1] - successes[i] for i in range(x - 1)])
    plt.subplot(3, 2, 3)
    m1 = 1 + math.floor(math.log2(N))
    plt.title("Task 2")
    plt.xlabel("Numbers of successes")
    plt.hist(success_nums, m1, density=True)
    plt.subplot(3, 2, 4)
    m2 = 1 + math.floor(math.log2(len(distances)))
    plt.xlabel("Distances")
    plt.hist(distances, m2, density=True)


def task3(lambda_, N):
    success_nums = []
    distances = []
    for _ in range(N):
        successes = []
        prev = 0
        r = np.random.exponential(1 / lambda_)
        while prev + r <= 1:
            distances.append(r)
            successes.append(prev + r)
            prev += r
            r = np.random.exponential(1 / lambda_)
        success_nums.append(len(successes))
    plt.subplot(3, 2, 5)
    m1 = 1 + math.floor(math.log2(N))
    plt.title("Task 3-4")
    plt.xlabel("Numbers of successes")
    plt.hist(success_nums, m1, density=True)
    plt.subplot(3, 2, 6)
    m2 = 1 + math.floor(math.log2(len(distances)))
    plt.xlabel("Distances")
    plt.hist(distances, m2, density=True)


def main():
    lambda_ = 17
    n = 100_000
    N = 100
    plt.suptitle("Poisson process")
    plt.figure(figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    task1(lambda_, n, N)
    task2(lambda_, N)
    task3(lambda_, N)
    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close()


main()
