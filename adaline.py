import math
import random


x_train = []
y_train = []
w = []
accepted_error = 0.3
learning_rate = 0.2


# def generate_points(points_am, x, y, x_cord, y_cord, function_type='and'):
#     y_to_add = 1
#     if function_type == 'and':
#         if x_cord != 1 or y_cord != 1:
#             y_to_add = -1
#     elif function_type == 'or':
#         if x_cord == 0 and y_cord == 0:
#             y_to_add = -1
#
#     for i in range(points_am):
#         if x_cord > 0 and y_cord > 0:
#             new_point = (random.uniform(x_cord-0.07, x_cord + 0.07), random.uniform(y_cord-0.07, y_cord + 0.07))
#         elif x_cord > 0:
#             new_point = (random.uniform(x_cord - 0.07, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
#         elif y_cord > 0:
#             new_point = (random.uniform(0, x_cord + 0.07), random.uniform(y_cord - 0.07, y_cord + 0.07))
#         else:
#             new_point = (random.uniform(0, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
#         x.append(new_point)
#         y.append(y_to_add)


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
    return bipolar_result(activation)


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
    # x_copy = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # for xp in x_copy:
    #     generate_points(points_am, x_train, y, xp[0], xp[1], function_type)

    x_train, y_train = generate_points(points_am, 0.1)

    w = [random.uniform(-1.0, 1.0) for i in range(len(x_train[0]) + 1)]
    # bias
    w[0] = random.uniform(-1.0, 1.0)
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
            epochs += 1
            current_mean_square_error = sum_error / len(x_train)
    print(f"EPOCHS: {epochs}, last error: {current_mean_square_error}")


if __name__ == '__main__':
    adaline_learn(50)
    x_test, y_test = generate_points(20, 0.1)

    correct = 0
    for i in range(len(x_test)):
        if predict(x_test[i]) == y_test[i]:
            correct += 1
    print(f"Correct: {correct} / {len(x_test)}")
