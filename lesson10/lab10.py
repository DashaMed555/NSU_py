import numpy as np

learning_rate = 0.7
epochs = 10000

np.random.seed(89798)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_of_sigmoid(x):
    return x * (1 - x)


class Neuron:
    def __init__(self):
        self._weights = np.random.uniform(size=(2, 1))
        self._bias = np.random.uniform()
        self._output = np.empty((4, 1))

    def forward(self, x):
        self._output = sigmoid(x @ self._weights + self._bias)

    def backward(self, x, loss_):
        d = loss_ * derivative_of_sigmoid(self._output)
        self._weights += learning_rate * x.T.dot(d)
        self._bias += learning_rate * np.sum(d, axis=0, keepdims=True)
        return d

    @property
    def weights(self):
        return self._weights

    @property
    def output(self):
        return self._output


class Model:
    def __init__(self):
        self._hidden_neuron1 = Neuron()
        self._hidden_neuron2 = Neuron()
        self._output_neuron = Neuron()

    def forward(self, x):
        self._hidden_neuron1.forward(x)
        self._hidden_neuron2.forward(x)
        self._output_neuron.forward(np.append(self._hidden_neuron1.output, self._hidden_neuron2.output, axis=1))
        return self._output_neuron.output

    def backward(self, x, loss_):
        layer = np.append(self._hidden_neuron1.output, self._hidden_neuron2.output, axis=1)
        d = self._output_neuron.backward(layer, loss_)
        w = self._output_neuron.weights - learning_rate * layer.T.dot(d)
        self._hidden_neuron1.backward(x, d.dot(w[0:1]))
        self._hidden_neuron2.backward(x, d.dot(w[1:2]))


def main():
    train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label = np.array([[0], [1], [1], [0]])
    model = Model()
    for _ in range(epochs):
        predicted = model.forward(train)
        err = label - predicted
        model.backward(train, err)
    predicted = model.forward(train)
    print(predicted)


main()
