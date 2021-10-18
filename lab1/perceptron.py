import math
import random
from timeit import default_timer as timer

x_train = []
y_train = []
w = []
alpha = 0.3
fi = 0.7

w_min = -0.7
w_max = 0.7


def activation_value(x, w):
    value_sum = 0
    for i in range(len(w)):
        value_sum += x[i]*w[i]
    return value_sum


def activation_value_with_bias(x, w):
    value_sum = w[0]
    for i in range(len(x)):
        value_sum += x[i]*w[i + 1]
    return value_sum


def unipolar_result(z, fi):
    if z > fi:
        return 1
    return 0


def unipolar_result_with_bias(z):
    if z >= 0:
        return 1
    return 0


def bipolar_result(z):
    if z >= 0:
        return 1
    return -1


def bipolar_result_with_bias(z):
    if z >= 0:
        return 1
    return -1


def error(y, z):
    return y - z


def square_error(y, x):
    temp = 0
    for i in range(len(w)):
        temp += w[i] * x[i]
    err = pow((y - temp), 2)
    return err


def predict(xp):
    global w
    activation = activation_value(xp, w)
    return activation


def generate_points(size, max_offset):
    x_data = []
    y_data = []
    y_val = 0
    for i in range(size):
        x_data.append((random.uniform(y_val - max_offset, y_val + max_offset),
                       random.uniform(y_val - max_offset, y_val + max_offset)))
        y_data.append(y_val)

        x_data.append((random.uniform(y_val - max_offset, y_val + max_offset),
                       random.uniform(1 - max_offset, 1 + max_offset)))
        y_data.append(y_val)

        x_data.append((random.uniform(1 - max_offset, 1 + max_offset),
                       random.uniform(y_val - max_offset, y_val + max_offset)))
        y_data.append(y_val)

        x_data.append(
            (random.uniform(1 - max_offset, 1 + max_offset), random.uniform(1 - max_offset, 1 + max_offset)))
        y_data.append(1)
    return x_data, y_data


def perceptron_learn(points_am, function_type='and'):
    global x_train, y_train, w
    x_train, y_train = generate_points(points_am, 0.1)
    w = [random.uniform(w_min, w_max) for i in range(len(x_train[0]) + 1)]
    w[0] = random.uniform(w_min, w_max)
    # w = [random.uniform(w_min, w_max) for i in range(len(x_train[0]))]
    epochs = 0
    error_occurred = True
    while error_occurred:
        error_occurred = False
        for i in range(len(x_train)):
            zk = activation_value_with_bias(x_train[i], w)
            res = unipolar_result_with_bias(zk)
            error_val = error(y_train[i], res)
            if error_val > 0:
                error_occurred = True
            w[0] = w[0] + alpha * error_val
            for j in range(len(x_train[i])):
                w[j + 1] = w[j + 1] + alpha * error_val * x_train[i][j]
        epochs += 1
    print(f"EPOCHS: {epochs}")


if __name__ == '__main__':
    start = timer()
    perceptron_learn(2000)
    end = timer()
    print(end-start)
    x_test, y_test = generate_points(20, 0.1)

    correct = 0
    for i in range(len(x_test)):
        if unipolar_result_with_bias(activation_value_with_bias(x_test[i], w)) == y_test[i]:
            correct += 1
    print(f"Correct: {correct} / {len(x_test)}")

