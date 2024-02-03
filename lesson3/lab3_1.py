class Worker:
    def __init__(self, name="Incognito", salary=1000, savings=0, function=None):
        self._name = name
        self._salary = salary
        self._savings = savings
        self._function = function

    @property
    def name(self):
        return self._name

    @property
    def salary(self):
        return self._salary

    @property
    def savings(self):
        return self._savings

    @property
    def function(self):
        return self._function

    def take_salary(self, salary):
        self._savings += salary


class DataScientist(Worker):
    def __init__(self, name="Incognito", salary=1000, savings=0, function=None):
        super().__init__(name, salary, savings, function)

    def do_work(self, filename1, filename2):
        matrices1 = self.read_matrices_from_file(filename1)
        matrices2 = self.read_matrices_from_file(filename2)
        iterations_num = min(len(matrices1), len(matrices2))
        for i in range(iterations_num):
            if self.check_matrices(matrices1[i], matrices2[i]):
                c = self.operate_with_matrices(matrices1[i], matrices2[i], self.function)
                self.print_matrices([c])
                print(f"{self.name} has done some more work.\n")
            else:
                print(f"{self.name}: Failed. The matrices{i + 1} are incorrect.\n")

    @staticmethod
    def operate_with_matrices(matrix_a, matrix_b, operation):
        matrix_c = []
        for r in range(len(matrix_a)):
            line = []
            for c in range(len(matrix_a[0])):
                line.append(operation(matrix_a[r][c], matrix_b[r][c]))
            matrix_c.append(line)
        return matrix_c

    @staticmethod
    def print_matrices(matrices):
        for it in range(len(matrices)):
            for i in range(len(matrices[it])):
                print(' '.join(map(str, matrices[it][i])))

    @staticmethod
    def check_matrices(matrix_a, matrix_b):
        len_a0 = len(matrix_a[0])
        len_b0 = len(matrix_b[0])
        return len(matrix_a) == len(matrix_b) and len_a0 == len_b0 and \
            all(len(row) == len_a0 for row in matrix_a) and \
            all(len(row) == len_b0 for row in matrix_b)

    @staticmethod
    def read_matrices_from_file(infile):
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


class Pupa(DataScientist):
    def __init__(self, salary=1500, savings=0):
        super().__init__("Pupa", salary, savings, self.plus)

    @staticmethod
    def plus(a, b):
        return a + b


class Lupa(DataScientist):
    def __init__(self, salary=1400, savings=0):
        super().__init__("Lupa", salary, savings, self.minus)

    @staticmethod
    def minus(a, b):
        return a - b


class Accountant(Worker):
    def __init__(self, name="Mary", salary=1000, savings=0):
        super().__init__(name, salary, savings, self.give_salary)

    def give_salary(self, worker):
        if isinstance(worker, Pupa) or isinstance(worker, Lupa):
            salary = worker.salary
            worker.take_salary(salary)
            print(f"{self.name} payed {salary}. Now {worker.name} has {worker.savings} money.\n")


def test():
    lupa = Lupa()
    pupa = Pupa()
    mary = Accountant()
    file1 = "input1.txt"
    file2 = "input2.txt"
    pupa.do_work(file1, file2)
    mary.give_salary(pupa)
    lupa.do_work(file1, file2)
    lupa.do_work(file2, file1)
    mary.give_salary(lupa)
    mary.give_salary(lupa)


test()
