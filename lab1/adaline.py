import math
import random
from timeit import default_timer as timer

x_train = []
y_train = []
w = []
accepted_error = 0.254
learning_rate = 0.001

w_min = -0.8
w_max = 0.8


def bipolar_result(z):
    if z >= 0:
        return 1
    return -1


def activation_value(x, w):
    value_sum = w[0]
    for i in range(len(x)):
        value_sum += x[i]*w[i + 1]
    return value_sum


def calculate_error(y, z):
    return y - z


def predict(xp):
    global w
    activation = activation_value(xp, w)
    return activation


def generate_points(size, max_offset):
    x_data = []
    y_data = []
    y_val = -1
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


def adaline_learn(points_am, function_type='and'):
    global x_train, y_train, w, accepted_error

    x_train, y_train = generate_points(points_am, 0.1)

    w = [random.uniform(w_min, w_max) for i in range(len(x_train[0]) + 1)]
    # bias
    w[0] = random.uniform(w_min, w_max)
    epochs = 0

    current_mean_square_error = math.inf
    while current_mean_square_error > accepted_error:
        sum_error = 0.0
        for i in range(len(x_train)):
            prediction = predict(x_train[i])
            err = calculate_error(y_train[i], prediction)
            error_squared = err ** 2
            sum_error += error_squared
            w[0] = w[0] + 2 * learning_rate * err
            for j in range(len(x_train[i])):
                w[j + 1] = w[j + 1] + 2 * learning_rate * err * x_train[i][j]
            current_mean_square_error = sum_error / len(x_train)
        epochs += 1
    print(f"Last error: {current_mean_square_error} {epochs}")


if __name__ == '__main__':
    start = timer()
    adaline_learn(2000)
    end = timer()
    print(end - start)
    x_test, y_test = generate_points(20, 0.1)

    correct = 0
    for i in range(len(x_test)):
        if bipolar_result(predict(x_test[i])) == y_test[i]:
            correct += 1
    print(f"Correct: {correct} / {len(x_test)}")
