from model.config import *
from model.network import *
from model.utils import *
from model.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

lstm_conf = LSTM_Config()
lstm_conf.update(use_previous_model=True,
                 label_name=['mid_price_delta'],
                 feature_name=['buy1', 'bc1', 'sale1', 'sc1', 'MACD_hist', 'MACD_DIF'],
                 training_set_proportion=0.8,
                 time_step=10,
                 epoch=30,
                 LSTM_neuron_num=[20, 20, 10]
                 )

# drop zero 72.5
# two class punish 82
# three class punish 67

# step 1: Get dataset (csv)
data = pd.read_csv(lstm_conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
mid_price = data['mid_price']
# price = data['price']
# true_2_5_min_mean_price = data['2.5min_mean_price']
feature_and_label_name = lstm_conf['feature_name']
feature_and_label_name.extend(lstm_conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
data = feature_normalize(data)
train_size = int(len(data) * lstm_conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
# price = price[train_size + data.shape[1] - lstm_conf['time_step'] + 3:len(data)]
# true_2_5_min_mean_price = true_2_5_min_mean_price[train_size + data.shape[1] - lstm_conf['time_step'] + 3:len(data)]
train_x, train_y = data_transform_lstm_30s(train, lstm_conf['time_step'])
test_x, test_y = data_transform_lstm_30s(test, lstm_conf['time_step'])

# step 4: Create and train model_weight
network = LSTMs(lstm_conf)
if lstm_conf['use_previous_model'] == 1:
    network.load(lstm_conf['load_file_name'])
elif lstm_conf['use_previous_model'] == 2:
    network.load(lstm_conf['save_file_name'])
    network.strong_train(train_x, train_y)
    network.save('strongtrain_test.h5')
else:
    network.train(train_x, train_y)
    network.save(lstm_conf['save_file_name'])

# step 5: Predict
train_pred = network.predict(train_x)
test_pred = network.predict(test_x)

# step 6: Evaluate
evaluator = Evaluator()
print('simple evaluation')

# method1
acc = evaluator.evaluate_trend_simple(y_true=train_y, y_pred=train_pred)
print(acc)
acc = evaluator.evaluate_trend_simple(y_true=test_y, y_pred=test_pred)
print(acc)

# method 2
acc_train_list = evaluator.evaluate_divided_trend(train_y, train_pred)
acc_test_list = evaluator.evaluate_divided_trend(test_y, test_pred)
print('acc_train_list = ' + str(acc_train_list))
print('acc_test_list = ' + str(acc_test_list))

# step 7: Plot
train_mid_price = mid_price[0:train_size]
test_mid_price = mid_price[train_size:len(mid_price)]
test_mid_price = test_mid_price[:-9]
y_true = np.add(test_mid_price, test_y)
test_pred = [value[0] for value in test_pred]
y_pred = np.add(test_mid_price, test_pred)
plot_regression(y_true, y_pred, sample_num=400, title=lstm_conf['label_name'][0])
