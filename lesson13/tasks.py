import gensim
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def task1():
    with open('book.txt', 'r') as f:
        data = f.read()
    data_cleaned = ''.join(letter if letter.isalpha() or letter.isspace() else '' for letter in data)
    sentences = [sentence.lower().split() for line in data_cleaned.splitlines() for sentence in line.split('.')]
    model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, window=5, epochs=100)
    print("Similar words:")
    for word in ['future', 'ai', 'car', 'virtual', 'robots']:
        most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in model.wv.most_similar(word)[:3])
        print('\t%s -> %s' % (word, most_similar))


def task2():
    class Model(nn.Module):
        def __init__(self, units=32):
            super().__init__()
            self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
            self.dense = nn.Linear(units, 3)
            self.relu = nn.ReLU()

        def forward(self, x_):
            h, _ = self.lstm1(x_)
            outs = []
            for i in range(h.shape[0]):
                outs.append(self.relu(self.dense(h[i])))
            out = torch.stack(outs, dim=0)
            return out

    df = pd.read_csv("production.csv")
    info = df.groupby('API')[['Liquid', 'Gas', 'Water']].apply(lambda df_: df_.reset_index(drop=True))
    columns_num = len(info.columns)
    df_prod = info.unstack()

    data = df_prod.values
    data = data.reshape((data.shape[0], -1, columns_num),  order='F')
    data = data / data.max()

    data_tr = data[:40]
    data_tst = data[40:]

    x_data = [data_tr[:, i:i+12] for i in range(11)]
    y_data = [data_tr[:, i+1:i+13] for i in range(11)]

    x_data = np.concatenate(x_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    tensor_x = torch.Tensor(x_data)
    tensor_y = torch.Tensor(y_data)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=16)

    model = Model()
    opt = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        num = 0
        for x_t, y_t in dataloader:
            opt.zero_grad()
            outputs = model(x_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            num += 1

        print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')
    print("Finished Training")

    x_tst = data_tst[:, :12]
    predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))

    for i in range(12):
        x = np.concatenate((x_tst[:, i:], predicts), axis=1)
        x_t = torch.from_numpy(x).float()
        predicted = model(x_t).detach().numpy()
        last_predicted = predicted[:, -1:]
        predicts = np.concatenate((predicts, last_predicted), axis=1)

    plt.figure(figsize=(10, 6))
    for api in range(4):
        plt.subplot(2, 2, api + 1)
        plt.plot(np.arange(x_tst.shape[1]), x_tst[api, :, 0], label="Actual")
        plt.plot(np.arange(predicts.shape[1]) + x_tst.shape[1], predicts[api, :, 0], label="Prediction")
        plt.legend()
    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close()


def task3():
    with open('book.txt', 'r') as f:
        data = f.read()
    data_cleaned = ''.join(letter if letter.isalpha() or letter.isspace() else '' for letter in data)
    sentence_len = 46
    sentences = np.array([list(' '.join(sentence.lower().split()))
                          for line in data_cleaned.splitlines()
                          for sentence in line.split('.') if len(' '.join(sentence.split())) == sentence_len])
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(sentences.reshape(-1, 1))
    array = np.array([encoder.transform(sentences[i].reshape(-1, 1)).toarray() for i in range(len(sentences))])
    return array, encoder


def task4(array, encoder):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, 1, batch_first=True)
            self.dense = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.soft_max = nn.Softmax(0)

        def forward(self, x_):
            h, _ = self.rnn(x_)
            outs = []
            for i in range(h.shape[0]):
                outs.append(self.soft_max(self.dense(h[i])))
            out = torch.stack(outs, dim=0)
            return out

    def decode(data):
        (n_sentences_, sentence_len_, _) = data.shape
        return [''.join(encoder.categories_[0][data.argmax(2)[sentence_num][symbol_num]]
                        for symbol_num in range(sentence_len_)) for sentence_num in range(n_sentences_)]

    (n_sentences, sentence_len, encoding_size) = array.shape
    model = RNN(encoding_size, encoding_size * 3, encoding_size)

    train = array[:9]
    test = array[9:]

    input_chars_num = 10
    x_data = [train[:, i:i + input_chars_num] for i in range(sentence_len - input_chars_num - 1)]
    y_data = [train[:, i + 1:i + input_chars_num + 1] for i in range(sentence_len - input_chars_num - 1)]

    x_data = np.concatenate(x_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    tensor_x = torch.Tensor(x_data)
    tensor_y = torch.Tensor(y_data)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=16)

    num_epochs = 1000
    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        num = 0
        for x_t, y_t in dataloader:
            opt.zero_grad()
            outputs = model(x_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            num += 1

        if (epoch + 1) % 100 == 0:
            print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')
    print("Finished Training")

    print("Examining test data: ")
    predicts = test[:, :input_chars_num]

    for i in range(sentence_len - input_chars_num - 1):
        x = predicts[:, i:i + input_chars_num]
        x_t = torch.from_numpy(x).float()
        predicted = model(x_t).detach().numpy()
        last_predicted = predicted[:, -1:]
        predicts = np.concatenate((predicts, last_predicted), axis=1)
    print("Actual sentences:\t ", decode(test))
    print("Predicted sentences: ", decode(predicts))

    print("Examining train data: ")
    predicts = train[:, :input_chars_num]

    for i in range(sentence_len - input_chars_num - 1):
        x = predicts[:, i:i + input_chars_num]
        x_t = torch.from_numpy(x).float()
        predicted = model(x_t).detach().numpy()
        last_predicted = predicted[:, -1:]
        predicts = np.concatenate((predicts, last_predicted), axis=1)
    print("Actual sentences:\t ", decode(train))
    print("Predicted sentences: ", decode(predicts))


def main():
    task1()
    task2()
    array, encoder = task3()
    task4(array, encoder)


main()
