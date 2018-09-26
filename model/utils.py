import numpy as np
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import csv
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from model.config import *
from tensorflow.python.ops import *
import seaborn as sns
import pandas as pd


def data_transform_lstm(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        x.append(data[i:time_step + i, 0:data.shape[1] - 1])
        y.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)

def data_transform_lstm2(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num - 1):
        x.append(data[i:time_step + i, 0:data.shape[1] - 1])
        y.append(data[time_step + i, -1])
    return np.array(x), np.array(y)


def data_transform_lstm_30s(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        window = data[i:time_step + i, 0:data.shape[1] - 1]
        window_mean_price = np.average(window[:, 0])
        x.append(data[i:time_step + i, 0:data.shape[1] - 1])
        y.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)


def data_transform_lstm_mv(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y1 = []
    y2 = []
    mid_price = []
    mean_price = []
    for i in range(window_num):
        window = data[i:time_step + i, 0:data.shape[1] - 4]
        x.append(window)
        y1.append(data[time_step + i - 1, -2])
        y2.append(data[time_step + i - 1, -1])
        mid_price.append(data[time_step + i - 1, -3])
        mean_price.append(data[time_step + i - 1, -4])
    return np.array(x), [np.array(y1), np.array(y2)], mid_price, mean_price


def data_transform_cnn(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        temp = data[i:time_step + i, 0:data.shape[1] - 1]
        temp = np.reshape(temp, (time_step, int((data.shape[1] - 1) / 2), 2))
        x.append(temp)
        y.append(np.sign(data[time_step + i - 1, -1]))
    return np.array(x), np.array(y)


def data_transform_for_xgboost(raw_data):
    data = np.array(raw_data)
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i, 0:data.shape[1] - 1])
        y.append(data[i, -1])
    return np.array(x), np.array(y)


def feature_normalize(data, label_num=1):
    scaler_feature = MinMaxScaler(feature_range=(0, 1))
    data[:, 0:-label_num] = scaler_feature.fit_transform(data[:, 0:-label_num])
    return data


def normalize(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data


def inverse(dataSet, scaler):
    return scaler.inverse_transform(dataSet)


def drop_zero(y_true, y_pred):
    y_true_ind = 100 * tf.abs(tf.clip_by_value(y_true, -0.01, 0.01))
    square = tf.square(y_true - y_pred)
    sum = K.mean(tf.multiply(y_true_ind, square))
    return sum


def two_class_penalty(y_true, y_pred):
    penalty = 5
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


def smooth(y_true, y_pred):
    diff = y_true - y_pred
    mse = K.mean(tf.square(diff))
    rs = bitwise_ops.right_shift(diff)
    grad = rs - diff
    rs = bitwise_ops.right_shift(grad)
    grad = rs - grad
    sum = K.mean(tf.square(grad))
    return sum + mse


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


def one_hot_decode(y):
    decode = []
    for vector in y:
        ind = vector.index(1)
        decode.append(ind)
    return decode


def batch_labelize_prob_vector(y):
    labelized = []
    for vector in y:
        vector = labelize_prob_vector(vector)
        labelized.append(vector)
    return labelized


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


def save_feature_selection(feature_list, acc):
    csv_file = open('feature_selection.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([acc] + feature_list)
    return


def over_sampling_naive(train_x, train_y):
    pos_indices = find_all_indices(train_y, 1)
    neg_indices = find_all_indices(train_y, -1)
    zero_indices = find_all_indices(train_y, 0)
    pos_sample = np.array(train_x)[pos_indices]
    neg_sample = np.array(train_x)[neg_indices]
    pos_size = len(pos_sample)
    neg_size = len(neg_sample)
    zero_size = len(train_x) - pos_size - neg_size
    power = zero_size / (pos_size + neg_size)

    pos_sample = _over_sampling_naive(pos_sample, power)
    neg_sample = _over_sampling_naive(neg_sample, power)

    train_x = list(np.array(train_x)[zero_indices])
    train_y = [0 for _ in range(zero_size)]
    train_x.extend(pos_sample)
    train_y.extend([1 for _ in range(len(pos_sample))])
    train_x.extend(neg_sample)
    train_y.extend([-1 for _ in range(len(neg_sample))])

    return np.array(train_x), np.array(train_y)


def over_sampling_smote(train_x, train_y):
    pos_indices = find_all_indices(train_y, 1)
    neg_indices = find_all_indices(train_y, -1)
    zero_indices = find_all_indices(train_y, 0)
    pos_sample = np.array(train_x)[pos_indices]
    neg_sample = np.array(train_x)[neg_indices]
    pos_size = len(pos_sample)
    neg_size = len(neg_sample)
    zero_size = len(train_x) - pos_size - neg_size
    power = zero_size / (pos_size + neg_size)

    pos_sample = _over_sampling_smote(pos_sample, power)
    neg_sample = _over_sampling_smote(neg_sample, power)

    train_x = list(np.array(train_x)[zero_indices])
    train_y = [0 for _ in range(zero_size)]
    train_x.extend(pos_sample)
    train_y.extend([1 for _ in range(len(pos_sample))])
    train_x.extend(neg_sample)
    train_y.extend([-1 for _ in range(len(neg_sample))])

    return np.array(train_x), np.array(train_y)


def _over_sampling_smote(sample, power):
    kdtree = KDTree(sample)
    indices = [i for i in range(len(sample))]
    np.random.shuffle(indices)
    new_sample_list = []
    count = int(power * len(sample)) - len(sample)
    each = int(power)
    feature_num = len(sample[0])

    for ori_ind in indices:
        _, near_ind = kdtree.query([sample[ori_ind]], each)
        for i in near_ind[0]:
            coef = np.random.rand()
            new_sample = [coef * sample[i][j] + (1 - coef) * sample[ori_ind][j] for j in range(feature_num)]
            new_sample_list.append(new_sample)
            count -= 1
        if count < 0:
            break

    sample = list(sample)
    sample.extend(new_sample_list)
    return sample


def _over_sampling_naive(sample, power):
    indices = [i for i in range(len(sample))]
    np.random.shuffle(indices)
    new_sample_list = []
    count = int(power * len(sample)) - len(sample)
    each = int(power)

    for ind in indices:
        for i in range(each):
            new_sample_list.append(sample[ind])
        count -= each
        if count < 0:
            break

    sample = list(sample)
    sample.extend(new_sample_list)
    return sample


def find_all_indices(data_list, value):
    indices = []
    for i in range(len(data_list)):
        if data_list[i] == value:
            indices.append(i)
    return indices


def show_feature_importance(clf, feature_list):
    fi_list = clf.feature_importances_
    ind = np.argsort(fi_list)
    feature_list = list(np.array(feature_list)[ind])
    fi_list = list(fi_list[ind])
    feature_list.reverse()
    fi_list.reverse()

    for i in range(len(ind)):
        print(feature_list[i], ': ', fi_list[i])
    np.save('feature_importance', [feature_list, fi_list])


def extract_feature_and_label(data, feature_name_list, label_name_list):
    feature_and_label_name = list(np.copy(feature_name_list))
    feature_and_label_name.extend(label_name_list)
    return data[feature_and_label_name].values


def divide_train_and_test(data, ratio):
    train_size = int(len(data) * ratio)
    train, test = data[0:train_size, :], data[train_size:len(data), :]
    return train, test


def bagging(*pred_list):
    bagging_pred = []
    for i in range(len(pred_list[0])):
        sum = 0
        for j in range(len(pred_list)):
            sum += pred_list[j][i]
        sum /= len(pred_list)
        bagging_pred.append(sum)
    return bagging_pred


def plot_scatter(y_true, y_pred, sample_size=50):
    sample_size = 50
    x_list = []
    pred_list = []
    true_list = []
    for i in range(sample_size):
        if y_true[i] != 0:
            x_list.append(i)
            pred_list.append(y_pred[i])
            true_list.append(y_true[i])
    try:
        plt.scatter(x_list, true_list)
        plt.scatter(x_list, pred_list, marker='x')
    except:
        print(x_list)
        print(true_list)
        print(pred_list)
    plt.show()


def plot_confidence_interval(true_mean_price, mean_list, std_list, sample_num=300):
    true_mean_price = true_mean_price[0:sample_num]
    mean_list = mean_list[0:sample_num]
    std_list = std_list[0:sample_num]
    plt.figure(figsize=(100, 15))
    up, down = mean_list + std_list * z_95, mean_list - std_list * z_95

    plt.plot(true_mean_price, color='g')
    plt.plot(mean_list, color='y')
    plt.plot(up, color='r')
    plt.plot(down, color='r')
    plt.legend(['true_mean_price', 'true_price', 'predict_mean', 'confidence interval'], loc='upper left')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()


def plot_classification(y_true, y_pred, sample_num=1000):
    y_true = y_true[0:sample_num]
    y_pred = y_pred[0:sample_num]
    correct = [y_true[i] ^ y_pred[i] for i in range(len(y_true))]
    height = np.random.random(len(y_true))
    dt = pd.DataFrame(data=list(zip(y_true, y_pred, correct, height)),
                      columns=['true', 'predicted', 'correct', 'height'])
    sns.stripplot(x='true', y='height', hue='predicted', data=dt, alpha=0.8)
    plt.title('XGBoost')
    plt.ylabel('')
    plt.yticks(range(1), [''])
    plt.xlabel('trend')
    plt.xticks(range(3), ['fall(-1)', 'unchanged', 'rise(1)'])
    plt.show()


def plot_regression(y_true, y_pred, sample_num=1000, title=''):
    start = 1000
    y_true = y_true[start:start + sample_num]
    y_pred = y_pred[start:start + sample_num]

    plt.figure(figsize=(12, 8))

    plt.plot(y_true)
    plt.plot(y_pred, color='green')
    plt.title(title)
    plt.legend(['true', 'predicted'])
    plt.show()
