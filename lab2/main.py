import math
import numpy as np
from random import uniform
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
import tensorflow as tf


class Mlp:
    def __init__(self, data_shape):
        self.weights = []
        self.biases = []
        self.layers = []
        self.data_shape = data_shape

    def add_layer(self, number_of_nodes, function):
        self.layers.append((number_of_nodes, function))
        if not self.weights:
            weights = np.zeros((number_of_nodes, self.data_shape[1]))
            self.biases.append(np.ones((number_of_nodes, self.data_shape[0])))
        else:
            weights = np.zeros((number_of_nodes, self.weights[-1].shape[0]))
            self.biases.append(np.ones((number_of_nodes, self.weights[-1].shape[1])))

        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights[i][j] = uniform(0, 1.0)

        self.weights.append(weights)

    @staticmethod
    def activate(x, function):
        if function == 'sigmoid':
            return Mlp.sigmoid(x)
        if function == 'tanh':
            return Mlp.tanh(x)
        if function == 'relu':
            return Mlp.relu(x)

    @staticmethod
    def sigmoid(x_data):
        fun = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
        return fun(x_data)

    @staticmethod
    def tanh(x_data):
        fun = np.vectorize(lambda x: (2 / (1 + math.exp(-2 * x))) + 1)
        return fun(x_data)

    @staticmethod
    def relu(x_data):
        fun = np.vectorize(lambda x: max(0, x))
        return fun(x_data)

    @staticmethod
    def softmax(x_data):
        total_val = np.sum(np.exp(x_data))
        fun = np.vectorize(lambda x: math.exp(x) / total_val)
        return fun(x_data)

    def forward(self, layer_inputs):
        outputs = [np.array(layer_inputs)]
        for i in range(len(self.layers) - 1):
            temp_matrix = np.dot(outputs[-1], self.weights[i].T)
            for element in temp_matrix:
                element += self.biases[i].T
            outputs.append(Mlp.activate(temp_matrix, self.layers[i][1]))

        return outputs

    def train(self, x_train, y_train):
        layer_outputs = self.forward(x_train)
        return Mlp.softmax(layer_outputs[-1])

    def fit(self):
        pass


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
    # show the figure
    pyplot.show()
    return x_train, y_train


def normalize_data(images, labels):
    images = np.array(images, dtype='float32')
    images /= 255

    labels = to_categorical(labels)

    return images, labels


if __name__ == '__main__':
    x_train, y_train = load_data()
    x, y = normalize_data(x_train, y_train)
    mlp = Mlp(x[0].shape)
    mlp.add_layer(10, 'relu')
    mlp.add_layer(5, 'relu')
    mlp.add_layer(3, 'relu')
    mlp.train(x, y)
