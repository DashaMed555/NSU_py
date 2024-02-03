import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("length", type=int, nargs=1, help="Length of some random list.")
args = parser.parse_args()
length = args.length[0]

list_ = []
for i in range(length):
    list_.append(random.random())

for n in range(length - 1):
    for i in range(length - 1):
        if list_[i] > list_[i + 1]:
            list_[i], list_[i + 1] = list_[i + 1], list_[i]

print(list_)
