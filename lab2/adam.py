
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import math
import random


class MLP:
    def __init__(self, beta1=0.9, beta2=0.9):
        self.l_rate = 0.05
        self.sizes = []
        self.activation_functions = []
        self.weights = []
        self.biases = []

        self.activated_layer_outputs = []
        self.layer_outputs = []

        self.w_acc = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = []
        self.v = []
        self.t = 1

    def add_layer(self, size, function):
        self.sizes.append(size)
        self.activation_functions.append(function)

    def add_output_layer(self):
        self.sizes.append(10)
        self.activation_functions.append('softmax')
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(len(self.sizes) - 1):
            self.weights.append(np.random.normal(scale=0.5, size=(self.sizes[i + 1], self.sizes[i])))
            self.biases.append(np.ones(self.sizes[i + 1]))

            self.w_acc = [w * 0 for w in self.weights]
            self.m = [w * 0 for w in self.weights]
            self.v = [w * 0 for w in self.weights]

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

    def update_parameters_batch(self, weight_updates, bias_updates, num_of_elems_in_batch):
        e = 0.00000001
        for i in range(len(self.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * weight_updates[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(weight_updates[i])

            m_W_dash = self.m[i] / (1 - np.power(self.beta1, self.t))
            v_W_dash = self.v[i] / (1 - np.power(self.beta2, self.t))

            self.weights[i] -= (self.l_rate * m_W_dash / (np.sqrt(v_W_dash) + e)) / num_of_elems_in_batch
            self.biases[i] -= self.l_rate * bias_updates[i] / num_of_elems_in_batch
        self.t += 1

    def train_batch(self, x_train, y_train, x_test, y_test, num_of_batches=1000, epochs=10):
        batches_x = np.array_split(x_train, num_of_batches)
        batches_y = np.array_split(y_train, num_of_batches)
        min_err = math.inf
        weight_cp = np.copy(self.weights)
        bias_cp = np.copy(self.biases)
        for i in range(epochs):
            for x_b, y_b in zip(batches_x, batches_y):
                weight_updates = None
                bias_updates = None
                for x, y in zip(x_b, y_b):
                    x = np.reshape(x, x.shape[0])
                    softed_output = self.go_forward(x)
                    weight_updates_val, bias_updates_val = self.go_backwards(y, softed_output)
                    if weight_updates is None and bias_updates is None:
                        weight_updates = weight_updates_val
                        bias_updates = bias_updates_val
                    else:
                        for j in range(len(weight_updates_val)):
                            bias_updates[j] += bias_updates_val[j]
                            weight_updates[j] += weight_updates_val[j]
                self.update_parameters_batch(weight_updates, bias_updates, len(x_b))
            print(f"Epoch {i}")
            accuracy = self.calculate_accuracy(x_test, y_test)
            print(f'Accuracy: {accuracy * 100}')
            if 1 - accuracy < min_err:
                min_err = 1 - accuracy
                weight_cp = np.copy(self.weights)
                bias_cp = np.copy(self.biases)
            else:
                print("Weights reset to previous values.")
                self.weights = weight_cp
                self.biases = bias_cp

    def train(self, x_train, y_train, x_val, y_val, epochs=10):
        for i in range(epochs):
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
    mlp.add_layer(128, 'sigmoid')
    mlp.add_layer(64, 'sigmoid')
    mlp.add_output_layer()
    mlp.train_batch(x_train, y_train, x_val, y_val, num_of_batches=1000, epochs=5)
    print('Start training')


if __name__ == '__main__':
    main()
