import numpy as np
import pandas as pd
from keras import backend as K


def data_transform(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        each = data[i:time_step + i, 0:data.shape[1] - 1]
        # each = np.reshape(each, (time_step, data.shape[1] - 1, 1))
        x.append(each)
        y.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)


def my_loss(y_true, y_pred):
    size = y_true.shape[0]
    sum = 0
    penalty = 5
    try:
        size = int(size)
        for i in range(size):
            if y_true[i][1] * y_pred[i][1] >= 0:
                sum += (y_true[i][1] - y_pred[i][1])**2
            else:
                sum += penalty * (y_true[i][1] - y_pred[i][1])**2
        return sum
    except:
        return K.mean(K.square(y_pred - y_true), axis=-1)


def one_hot_encode(y, category_num):
    encode = []
    value_list = []
    for value in y:
        vector = [0 for _ in range(category_num)]
        if value not in value_list:
            value_list.append(value)
        ind = value_list.index(value)
        try:
            vector[ind] = 1
        except IndexError:
            print('Error: index = ' + str(ind) + ' value = ' + str(value))
        encode.append(vector)
    return encode


def labelize_prob_vector(vector):
    vector = list(vector)
    size = len(vector)
    m = max(vector)
    ind = vector.index(m)
    label_vec = [0 for _ in range(size)]
    label_vec[ind] = 1
    return label_vec


def sign(vector):
    sign_vector = []
    for value in vector:
        if value > 0:
            sign_vector.append(1)
        if value == 0:
            sign_vector.append(0)
        if value < 0:
            sign_vector.append(-1)
    return sign_vector

