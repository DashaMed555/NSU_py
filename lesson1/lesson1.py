import random
import math
import os

print('task 1')
a = str(random.randint(100, 999))
print(a)
s = 0
for i in range(len(a)):
    s += int(a[i])
print(s, '\n')


print('task 2')
a = str(random.randint(0, 2 ** 31))
print(a)
s = 0
for i in range(len(a)):
    s += int(a[i])
print(s, '\n')


print('task 3')
r = float(input('Радиус сферы: '))
square = 4 * math.pi * r ** 2
volume = 4 / 3 * math.pi * r ** 3
print(f'Площадь поверхности сферы = {square}')
print(f"Объём сферы = {volume}", '\n')


print('task 4')
year = int(input('Год: '))
if year % 4 == 0 and (year % 100 == 0 and year % 400 == 0 or year % 100 != 0):
    print('Високосный')
else:
    print('Не високосный')
print()


print('task 5')
N = int(input('N = '))
lis = []
for i in range(2, N + 1):
    f = 0
    for j in range(2, i):
        if i % j == 0:
            f = 1
            break
    if f == 0:
        lis.append(i)
print(f'Список простых чисел: {lis}')
print()


print('task 6')
X = float(input('Сумма вклада: '))
Y = int(input('Количество лет: '))
money = X
for i in range(Y):
    money *= 1.1
print(f'Через указанный срок на счету будет {money:.2f} рублей')
print()

print('task 7')
name_dir = input('Имя папки: ')
tree = os.walk(name_dir)
for i in tree:
    for k in range(len(i[2])):
        print(i[0] + '/' + i[2][k])
