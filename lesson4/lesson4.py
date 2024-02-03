import numpy as np

# Task 1

print("Task 1:\n")

arr = np.random.randint(100, 108, size=16)
print("Array:\t\t\t\t", arr)

unique_values, frequency = np.unique(arr, return_counts=True)
print("Unique values:\t\t", unique_values)
print("Frequency:\t\t\t", frequency)

sorting_indices = np.argsort(frequency)[::-1]
print("Sorting indices:\t", sorting_indices)
print("Sorted values:\t\t", unique_values[sorting_indices])
print("\n\n")

# Task 2

print("Task 2:\n")

h = 10
w = 15
picture = np.random.randint(0, 255, (h, w), np.uint8)
print("Picture:\n", picture)
unique_colors = np.unique(picture)
print("\nUnique colors: ", unique_colors)
print("\n\n")

# Task 3

print("Task 3:\n")


def sailing_avg(vector_, n):
    sums = np.cumsum(np.insert(vector_, 0, 0))
    return (sums[n:] - sums[:-n]) / n


vector = np.random.randn(10)
print("Vector: ", vector)
print("Sailing average: ", sailing_avg(vector, 4))
print("\n\n")

# Task 34

print("Task 4:\n")

matrix = np.random.randint(1, 15, (20, 3))
print("Matrix:\n", matrix)
conditions = [a + b > c and a + c > b and b + c > a for (a, b, c) in matrix]
print("\nSides of a triangle:\n", matrix[conditions])
