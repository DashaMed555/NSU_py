import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, 32)
        self.fc2 = nn.Linear(32, out_ch, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        y = self.relu(h)
        return y


def task1(in_ch, out_ch):
    model = nn.Sequential(
        nn.Linear(in_ch, 32),
        nn.ReLU(),
        nn.Linear(32, out_ch, bias=False),
        nn.ReLU()
    )
    return model


def task2():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(256, 64)
            self.l2 = nn.Linear(64, 16)
            self.l3 = nn.Linear(16, 4)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(0)

        def forward(self, x):
            res = self.l1(x)
            act = self.relu(res)
            res = self.l2(act)
            act = self.tanh(res)
            res = self.l3(act)
            act = self.softmax(res)
            return act
    return Model()


def task3():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Conv2d(3, 8, 2)
            self.l2 = nn.Conv2d(8, 16, 2)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            res = self.l1(x)
            act = self.relu(res)
            pooled = self.max_pool(act)
            res = self.l2(pooled)
            act = self.relu(res)
            pooled = self.max_pool(act)
            return pooled
    return Model()


def task4():
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    picture = torch.Tensor(3, 19, 19).random_(0, 255)
    model = nn.Sequential(
        task3(),
        Lambda(torch.ravel),
        task2()
    )
    res = model(picture)
    return res


def main():
    model1 = task1(64, 10)
    res = task4()


main()
