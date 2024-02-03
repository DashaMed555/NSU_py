import argparse
import numpy as np


def read_args():
    parser = argparse.ArgumentParser(description="Training ML-model...")

    parser.add_argument("infile_r", nargs=1, type=str, help="A file containing real data.")
    parser.add_argument("infile_s", nargs=1, type=str, help="A file containing synthetic data.")
    parser.add_argument("possibility", nargs=1, type=float, help="The possibility of choice of synthetic data.")

    args = parser.parse_args()
    file_r = args.infile_r[0]
    file_s = args.infile_s[0]
    p = args.possibility[0]
    return file_r, file_s, p


def read_data(file):
    try:
        with open(file) as f:
            data = np.fromiter(map(int, f.readline().split()), int)
    except OSError:
        print("OS error: ", OSError)
    return data


def method1(x, y, p_):
    xy = np.stack((x, y))
    x_bool = np.random.random(np.size(x)) > p_
    y_bool = ~x_bool
    xy_bool = np.stack((x_bool, y_bool))
    xy = np.transpose(xy, (1, 0)).ravel()
    xy_bool = np.transpose(xy_bool, (1, 0)).ravel()
    return xy[xy_bool]


def method2(x, y, p_):
    return np.where(np.random.random(np.size(x)) > p_, x, y)


def main():
    file_r, file_s, p = read_args()
    try:
        data_r = read_data(file_r)
    except ValueError:
        return
    try:
        data_s = read_data(file_s)
    except ValueError:
        return
    if 0 <= p <= 1:
        data_r_shape = data_r.shape
        data_s_shape = data_s.shape
        if data_r_shape == data_s_shape:
            print("1. Training data:\n", method1(data_r, data_s, p))
            print("\n\n2. Training data:\n", method2(data_r, data_s, p))
            print("\n-----------------------------------------------\n")
        else:
            print("Different sizes of data.")
    else:
        print("Invalid possibility.")


main()
main()
main()
