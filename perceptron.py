import math
import random

x_train = []
y_train = []
w = [0.1, 0.1]
alpha = 0.3
fi = 0.7

# Adaline
mi = 0.3
accepted_error = 0.3


def activation_value(x, w):
    value_sum = 0
    for i in range(len(w)):
        value_sum += x[i]*w[i]
    return value_sum


def unipolar_result(z, fi):
    if z > fi:
        return 1
    return 0


def bipolar_result(z, fi):
    if z > fi:
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


def generate_points(points_am, x, y, x_cord, y_cord, function_type='and'):
    y_to_add = 1
    if function_type == 'and':
        if x_cord != 1 or y_cord != 1:
            y_to_add = 0
    elif function_type == 'or':
        if x_cord == 0 and y_cord == 0:
            y_to_add = 0

    for i in range(points_am):
        if x_cord > 0 and y_cord > 0:
            new_point = (random.uniform(x_cord-0.07, x_cord + 0.07), random.uniform(y_cord-0.07, y_cord + 0.07))
        elif x_cord > 0:
            new_point = (random.uniform(x_cord - 0.07, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
        elif y_cord > 0:
            new_point = (random.uniform(0, x_cord + 0.07), random.uniform(y_cord - 0.07, y_cord + 0.07))
        else:
            new_point = (random.uniform(0, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
        x.append(new_point)
        y.append(y_to_add)


def perceptron_learn(points_am, function_type='and'):
    x_copy = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for xp in x_copy:
        generate_points(points_am, x_train, y_train, xp[0], xp[1], function_type)

    error_occurred = True
    while error_occurred:
        error_occurred = False
        for i in range(len(x_train)):
            zk = activation_value(x_train[i], w)
            res = unipolar_result(zk, fi)
            error_val = error(y_train[i], res)
            if error_val > 0:
                error_occurred = True
            for j in range(len(w)):
                w[j] = w[j] + alpha * error_val * x_train[i][j]


def adaline_learn(points_am, function_type='and'):
    x_copy = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for xp in x_copy:
        generate_points(points_am, x_train, y_train, xp[0], xp[1], function_type)

    current_mean_square_error = math.inf
    temp_errors = []
    while current_mean_square_error > accepted_error:
        temp_errors.clear()
        for i in range(len(x_train)):
            square_err = square_error(y_train[i], x_train[i])
            temp_errors.append(square_err)
            for j in range(len(w)):
                w[j] = w[j] + (mi * x_train[i][j] * math.sqrt(square_err))
        current_mean_square_error = sum(temp_errors)/len(temp_errors)
        print(current_mean_square_error)


if __name__ == '__main__':
    # perceptron_learn(10)
    # val = activation_value((0.999, 0.987), w)
    # res = unipolar_result(val, fi)
    # print(res)
    adaline_learn(10)
