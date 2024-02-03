import argparse

parser = argparse.ArgumentParser()
parser.add_argument("height", type=int, nargs=1, help="Height of Pascal's Triangle")
args = parser.parse_args()
height = args.height[0]

triangle = [[0 for _ in range(height)] for _ in range(height)]
for i in range(height):
    triangle[0][i] = 1
for i in range(1, height):
    triangle[i][0] = 1
    for j in range(1, height - i):
        triangle[i][j] = triangle[i][j - 1] + triangle[i - 1][j]

for i in range(height - 1, -1, -1):
    for j in range(i):
        print('\t', end='')
    h = height - i - 1
    for j in range(height - i):
        print(str(triangle[h][j]) + '\t\t', end='')
        h -= 1
    print()
