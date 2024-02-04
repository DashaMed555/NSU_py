import numpy as np
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    train_dataset = torchvision.datasets.MNIST(
        root="MNIST/train", train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False)

    test_dataset = torchvision.datasets.MNIST(
        root="MNIST/test", train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False)

    train = transform_data(train_dataset)
    test = transform_data(test_dataset)

    weights, samples = average_digits_and_samples(train)
    accuracy, answers, real_digits = get_accuracy_and_answers_and_real_digits(test, weights, -100)
    answers = answers[:300]
    real_digits = real_digits[:300]
    print("Accuracy: ", accuracy)

    model = TSNE(n_components=2, method="exact")
    points = model.fit_transform(samples.reshape((300, 784)))
    plt.figure(figsize=(6, 6))
    for digit in range(10):
        range_l = digit * 30
        range_r = (digit + 1) * 30 + 1
        plt.scatter(points[range_l:range_r, 0], points[range_l:range_r, 1], label=f"{digit + 1}")
    plt.legend()
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()

    points = model.fit_transform(answers)
    plt.figure(figsize=(6, 6))
    for digit in range(10):
        plt.scatter(points[:, 0][real_digits == digit], points[:, 1][real_digits == digit], label=f"{digit + 1}")
    plt.legend(loc='center right')
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()


def encode_label(label):
    encoding = np.zeros((10, 1))
    encoding[label] = 1.
    return encoding


def transform_data(data):
    features = [np.reshape(x[0][0].numpy(), (784, 1)) for x in data]
    labels = [encode_label(x[1]) for x in data]
    return list(zip(features, labels))


def average_digits_and_samples(data):
    digits = np.empty(10, 'object')
    samples = np.empty((10, 30, 784, 1))
    for digit in range(10):
        digits[digit] = []
    for image in data:
        digits[np.argmax(image[1])].append(image[0])
    for digit in range(10):
        samples[digit] = np.asarray(digits[digit][:30])
    for digit in range(10):
        digits[digit] = np.average(np.asarray(digits[digit]), axis=0)
    digits = [np.transpose(weight) for weight in digits]
    return digits, samples


def predict_digit(image, weights, b):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    return [sigmoid(np.dot(weight, image)[0][0] + b) for weight in weights]


def get_accuracy_and_answers_and_real_digits(data, weights, b):
    right_answers = 0
    answers = np.empty((10000, 10))
    real_digits = np.empty(10000)
    i = 0
    for image in data:
        answers[i] = predict_digit(image[0], weights, b)
        real_digits[i] = np.argmax(image[1])
        right_answers += np.argmax(image[1]) == np.argmax(answers[i])
        i += 1
    return right_answers / len(data), answers, real_digits


main()
