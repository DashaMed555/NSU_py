import random

# Task 1

print("Task 1:")
line = input("Введите строку: ").lower().replace(' ', '')
if line == line[::-1]:
    print("Эта строка - палиндром!")
else:
    print("Эта строка не является палиндромом.")


# Task 2

print("Task 2:")
line = input("Введите строку: ")
arr = line.split(' ')
arr.sort(key=len)
the_longest_word = arr[-1]
print(f'Самое длинное слово в этой строке: {the_longest_word}')

# Task 3

print("Task 3:")
N = int(input("Сколько сгенерировать чисел? Введите N: "))
rand_list = [random.randint(-2**7, 2**7 - 1) for i in range(N)]
print(f"Полученный список: {rand_list}")
odd_cnt = 0
even_cnt = 0
for i in rand_list:
    if i % 2 == 0:
        even_cnt += 1
    else:
        odd_cnt += 1
print(f"В нём {even_cnt} чётных чисел и {odd_cnt} нечётных.")


# Task 4

dictionary = {"Абсолютный": "Совершенный",
              "Актуальный": "Злободневный",
              "Анализ": "Разбор",
              "Габариты": "Размеры",
              "Метод": "Приём",
              "Натуральный": "Естественный"}
news = "Абсолютный метод позволил провести актуальный в эти дни анализ. \n" \
       "Удалось выяснить, что натуральный арбуз, как правило, имеет меньшие габариты."
print("Task 4:")
print(news)
news_upd = news
for key in dictionary.keys():
    up = news_upd.find(key)
    while up != -1:
        news_upd = news_upd.replace(key, dictionary[key])
        up = news_upd.find(key)
    low = news_upd.find(key.lower())
    while low != -1:
        news_upd = news_upd.replace(key.lower(), dictionary[key].lower())
        low = news_upd.find(key.lower())
print(news_upd)


# Task 5

fib_dict = {0: 0, 1: 1}


def fib(n):
    if n in fib_dict:
        return fib_dict[n]
    if n < 0:
        return -1
    fib_dict[n] = fib(n - 2) + fib(n - 1)
    return fib_dict[n]


print("Task 5:")
print([fib(n) for n in range(30)])


# Task 6

print("Task 6:")
with open('input1.txt') as f:
    content = f.read()

lines_cnt = len(content.split('\n'))
words_cnt = len(content.replace('\n', ' ').split())
symbols_cnt = len(content.replace('\n', '').replace(' ', ''))
print(f"В файле {lines_cnt} строк, {words_cnt} слов, {symbols_cnt} символов без пробелов.")


# Task 7

def geom_progression(b, q):
    while 1:
        b *= q
        yield b


print("Task 7:")
g_p = geom_progression(1, 0.5)
for i in range(5):
    print(next(g_p))
