import argparse


def main():
    """
    Reads infile and outfile from command line, reads integer matrices from infile,
    multiplies the matrices and writes the result to outfile.

    :return: nothing or raised errors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, nargs=1, help="Path to the file with matrices.")
    parser.add_argument("outfile", type=str, nargs=1, help="Path to the file with result.")
    args = parser.parse_args()
    try:
        matrices = read_matrices_from_file(args.infile[0])
    except ValueError:
        return
    len_a0 = len(matrices[0][0])
    len_b0 = len(matrices[1][0])
    len_b = len(matrices[1])
    if len_a0 == len_b and all(len(a) == len_a0 for a in matrices[0]) and all(len(b) == len_b0 for b in matrices[1]):
        matrix_c = multiply_matrices([matrices[0], matrices[1]])
        write_matrices_to_file([matrix_c], args.outfile[0])
    else:
        print("Such matrices cannot be multiplied.")


def read_matrices_from_file(infile):
    """
    Reads matrices from infile and returns them.

    Parameters
    ----------
    :param infile: a file from which the matrices will be read.

    :return: a list of read matrices.
    """
    matrices = [[]]
    try:
        with open(infile) as f:
            flag = 0
            it = 0
            for line in f:
                if line.replace(' ', '').replace('\n', '') != '':
                    flag = 1
                    matrices[it].append(list(map(int, line.split())))
                elif flag:
                    flag = 0
                    it += 1
                    matrices.append([])
    except OSError:
        print("OS error: ", OSError)
    return matrices


def multiply_matrices(matrices):
    """
    Multiplies matrices and returns the result of the multiplication.

    Parameters
    ----------
    :param matrices: a list of matrices that will be multiplied among themselves
                     in the order in which they appear in the list.

    :return: matrix c, which is the result of multiplying all matrices in the list.
    """
    c = []
    for it in range(len(matrices) - 1):
        if not c:
            a = matrices[it]
        else:
            a = c
        b = matrices[it + 1]
        len_a = len(a)
        len_b = len(b)
        len_b0 = len(b[0])
        for i in range(len_a):
            line = []
            for j in range(len_b0):
                s = 0
                for k in range(len_b):
                    s += a[i][k] * b[k][j]
                line.append(s)
            c.append(line)
    return c


def write_matrices_to_file(matrices, outfile):
    """
    Writes matrices to outfile and raises OS error if it cannot open outfile.

    Parameters
    ----------
    :param matrices: a list of matrices that will be written to outfile.
    :param outfile: a file into which the matrices will be written.

    :return: nothing or raised error.
    """
    try:
        with open(outfile, 'w') as f:
            for it in range(len(matrices)):
                for i in range(len(matrices[it])):
                    f.write(' '.join(map(str, matrices[it][i])))
                    f.write('\n')
    except OSError:
        print("OS error: ", OSError)


main()
