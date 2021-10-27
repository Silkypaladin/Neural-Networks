
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self):
        self.l_rate = 0.001
        self.sizes = []
        self.activation_functions = []
        self.weights = []
        self.biases = []

        self.activated_layer_outputs = []
        self.layer_outputs = []

    def add_layer(self, size, function):
        self.sizes.append(size)
        self.activation_functions.append(function)

    def add_output_layer(self):
        self.sizes.append(10)
        self.activation_functions.append('softmax')
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(len(self.sizes) - 1):
            self.weights.append(np.random.randn(self.sizes[i + 1], self.sizes[i]))
            self.biases.append(np.ones(self.sizes[i + 1]))

    @staticmethod
    def sigmoid(x, derivative):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activate(x, function, derivative=False):
        if function == 'sigmoid':
            return MLP.sigmoid(x, derivative)
        if function == 'tanh':
            return MLP.tanh(x, derivative)
        if function == 'relu':
            return MLP.relu(x, derivative)

    @staticmethod
    def relu(x, derivative):
        if derivative:
            return 1 / (1 + np.exp(-x))
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x, derivative):
        if derivative:
            return 1 - (MLP.tanh(x, False) ** 2)
        return (2 / (1 + np.exp(-2 * x))) + 1

    @staticmethod
    def softmax(x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def go_forward(self, x_train):
        self.activated_layer_outputs = [x_train]
        self.layer_outputs = []
        for i in range(len(self.sizes) - 1):
            out = np.dot(np.array(self.weights[i]), self.activated_layer_outputs[-1]) + self.biases[i]
            self.layer_outputs.append(out)

            if i < len(self.sizes) - 2:
                out_activated = MLP.activate(out, self.activation_functions[i])
            else:
                out_activated = MLP.softmax(out)
            self.activated_layer_outputs.append(out_activated)

        return self.activated_layer_outputs[-1]

    def go_backwards(self, y_train, out):
        layer_outputs_rev = self.layer_outputs[::-1]
        activated_outputs_rev = self.activated_layer_outputs[::-1]
        weights = self.weights[::-1]
        functions_rev = self.activation_functions[::-1]

        error = (out - y_train) * self.softmax(layer_outputs_rev[0], True)
        weight_updates = [np.outer(error, activated_outputs_rev[1])]
        bias_updates = [error]

        for i in range(len(self.weights) - 1):
            error = np.dot(weights[i].T, error) * MLP.activate(layer_outputs_rev[i + 1], functions_rev[i + 1], True)
            bias_updates.append(error)
            weight_updates.append(np.outer(error, activated_outputs_rev[i + 2]))

        return weight_updates[::-1], bias_updates[::-1]

    def update_all_parameters(self, w_m, b_m):
        for i in range(len(self.weights)):
            self.weights[i] -= self.l_rate * w_m[i]
            self.biases[i] -= self.l_rate * b_m[i]

    def calculate_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.go_forward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        for x, y in zip(x_train, y_train):
            x = np.reshape(x, x.shape[0])
            out = self.go_forward(x)  # y^ (wart. predykowana)
            w_update, b_update = self.go_backwards(y, out)
            self.update_all_parameters(w_update, b_update)

        accuracy = self.calculate_accuracy(x_val, y_val)
        print(f'Accuracy: {accuracy * 100}')


def main():
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    x = np.array(x, dtype='float32')
    x /= 255
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    mlp = MLP()
    mlp.add_layer(784, 'sigmoid')
    mlp.add_layer(196, 'sigmoid')
    mlp.add_layer(64, 'sigmoid')
    mlp.add_output_layer()
    mlp.train(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()
