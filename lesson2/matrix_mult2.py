import argparse


def main():
    """
    Reads infile and outfile from command line, reads integer matrices from infile,
    finds the convolution of the matrices and writes the result to outfile.

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
    len_a = len(matrices[0])
    len_a0 = len(matrices[0][0])
    len_b = len(matrices[1])
    len_b0 = len(matrices[1][0])
    if len_a >= len_b and len_a0 >= len_b0 and \
            all(len(a) == len_a0 for a in matrices[0]) and all(len(b) == len_b0 for b in matrices[1]):
        matrix_c = convolution(matrices[0], matrices[1])
        write_matrices_to_file([matrix_c], args.outfile[0])
    else:
        print("Convolution cannot be applied.")


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


def convolution(matrix, kernel):
    """
    Finds the convolution of the matrix with the kernel.

    :param matrix: the matrix relative to which the convolution will be found.
    :param kernel: convolution kernel.
    :return: matrix c that is the result of convolution.
    """
    len_a = len(matrix)
    len_a0 = len(matrix[0])
    len_b = len(kernel)
    len_b0 = len(kernel[0])
    c = []
    for i in range(len_a - len_b + 1):
        line = []
        for j in range(len_a0 - len_b0 + 1):
            s = 0
            for k in range(len_b):
                for n in range(len_b0):
                    s += matrix[i + k][j + n] * kernel[k][n]
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
