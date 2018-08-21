import numpy as np
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import xgboost as xgb


def data_transform_lstm(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        x.append(data[i:time_step + i, 0:data.shape[1] - 1])
        y.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)


def data_transform_cnn(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        temp = data[i:time_step + i, 0:data.shape[1] - 1]
        temp = np.reshape(temp, (time_step, int((data.shape[1] - 1) / 2), 2))
        x.append(temp)
        y.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)


def data_transform_for_xgboost(raw_data):
    data = np.array(raw_data)
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i, 0:data.shape[1] - 1])
        y.append(data[i, -1])
    return np.array(x), np.array(y)


def feature_normalize(data):
    scaler = MinMaxScaler()
    data[:, 0:-1] = scaler.fit_transform(data[:, 0:-1])
    return data


def drop_zero(y_true, y_pred):
    y_true_ind = 100 * tf.abs(tf.clip_by_value(y_true, -0.01, 0.01))
    square = tf.square(y_true - y_pred)
    sum = K.mean(tf.multiply(y_true_ind, square))
    return sum


def two_class_penalty(y_true, y_pred):
    penalty = 2
    penalty_ind = tf.abs(tf.clip_by_value(tf.sign(tf.multiply(y_true, y_pred)), -1, 0))
    normal_ind = tf.clip_by_value(tf.sign(tf.multiply(y_true, y_pred)), 0, 1)
    penalty_coef = penalty * penalty_ind + normal_ind
    square = tf.square(y_true - y_pred)
    sum = K.mean(tf.multiply(penalty_coef, square))
    return sum


# acc: p=1(MSE): 0.74  p=2: 0.80,0.74,0.78  p=3: 0.75,0.78,0.78,0.75  p=4: 0.76  p=5: 0.75  p=10: 0.69
def three_class_penalty(y_true, y_pred):
    penalty_1 = 2
    penalty_2 = penalty_1 * 1.5
    y_true_round = tf.round(100 * y_true)
    y_true_round = tf.clip_by_value(y_true_round, -1, 1)
    y_pred_round = tf.round(100 * y_pred)
    y_pred_round = tf.clip_by_value(y_pred_round, -1, 1)
    penalty_delta = tf.abs(y_true_round - y_pred_round)  # tensor中的元素的取值为0，1，2
    temp_0 = tf.abs(tf.clip_by_value(penalty_delta, 0, 1) - 1)
    temp_1 = penalty_1 * (tf.clip_by_value(penalty_delta, 0, 1))
    temp_2 = penalty_2 * (tf.clip_by_value(penalty_delta, 1, 2) - 1)
    coef = temp_0 + temp_1 + temp_2
    square = tf.square(y_true - y_pred)
    sum = K.mean(tf.multiply(coef, square))
    return sum


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
