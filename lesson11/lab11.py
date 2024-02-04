import torch


class Layer:
    def __init__(self, in_size, out_size, activation):
        self.weights = torch.Tensor(in_size, out_size).uniform_(-1, 1)
        self.bias = torch.Tensor(out_size).uniform_(-1, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(x @ self.weights + self.bias)


class Model:
    def __init__(self):
        super().__init__()
        self.l1 = Layer(256, 64, torch.nn.ReLU())
        self.l2 = Layer(64, 16, torch.nn.Tanh())
        self.l3 = Layer(16, 4, torch.nn.Softmax(0))

    def forward(self, x):
        res = self.l1.forward(x)
        res = self.l2.forward(res)
        res = self.l3.forward(res)
        return res
