# Task 1

class Item:
    def __init__(self, count=3, max_count=16):
        self.count = count
        self.max_count = max_count

    def update_count(self, val):
        if 0 <= val <= self.max_count:
            self.count = val
            return True
        elif 0 <= val:
            self.count = self.max_count
            return False
        else:
            self.count = 0

    def __add__(self, num):
        return self.count + num

    def __sub__(self, num):
        return self.count - num

    def __mul__(self, num):
        return self.count * num

    def __truediv__(self, num):
        if num:
            return self.count / num
        else:
            return num

    def __floordiv__(self, num):
        if num:
            return self.count // num
        else:
            return num

# Task 2

    def __gt__(self, num):
        return self.count > num

    def __ge__(self, num):
        return self.count >= num

    def __lt__(self, num):
        return self.count < num

    def __le__(self, num):
        return self.count <= num

    def __eq__(self, num):
        return self.count == num

    def __iadd__(self, num):
        res = self.count + num
        self.update_count(res)
        return self

    def __isub__(self, num):
        res = self.count - num
        self.update_count(res)
        return self

    def __imul__(self, num):
        res = self.count * num
        self.update_count(res)
        return self


# Task 3


class Fruit(Item):
    def __init__(self, ripe=True, eatable=True, **kwargs):
        super().__init__(**kwargs)
        self.ripe = ripe
        self.eatable = eatable


class Banana(Fruit):
    def __init__(self, color='yellow', origin='Turkey', **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.origin = origin


class Apple(Fruit):
    def __init__(self, color='red', sourness=3, **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.sourness = sourness


class Nuts(Item):
    def __init__(self, hardness=15, eatable=True, **kwargs):
        super().__init__(**kwargs)
        self.hardness = hardness
        self.eatable = eatable


class Coconut(Nuts):
    def __init__(self, size=30, origin='Thailand', **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.origin = origin


class Seed(Nuts):
    def __init__(self, fried=True, salty=True, **kwargs):
        super().__init__(**kwargs)
        self.fried = fried
        self.salty = salty


# Task 4

class Inventory:
    def __init__(self, size=10):
        self.size = size
        self.list = [None for _ in range(size)]

    def __getitem__(self, index):
        return self.list[index]

    def __setitem__(self, index, item):
        if (isinstance(item, Fruit) or isinstance(item, Nuts)) and \
                item.eatable and item.count != 0 and not self.list[index]:
            self.list[index] = item

    def take_items(self, index, count):
        if self.list[index]:
            new_count = self.list[index] - count
            self.list[index].update_count(new_count)
            if self.list[index] == 0:
                self.list[index] = None


apple = Apple()
coconut = Coconut()
inventory = Inventory()
print(inventory.list)
inventory[6] = apple
print(inventory[6])
print(inventory.list)
inventory.take_items(6, 3)
print(inventory.list)
