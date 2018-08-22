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
lstm_conf.update(use_previous_model=False,
                 load_file_name='lstm.h5')

# step 1: Get dataset (csv)
data = pd.read_csv(lstm_conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
data = extract_feature_and_label(data, lstm_conf['feature_name'], lstm_conf['label_name'])

# step 3: Preprocess
train, test = divide_train_and_test(data, lstm_conf['training_set_proportion'])
train_x, train_y = data_transform_lstm(train, lstm_conf['time_step'])
test_x, test_y = data_transform_lstm(test, lstm_conf['time_step'])

# step 4: Create and train model_weight
network = LSTMs(lstm_conf)
if lstm_conf['use_previous_model']:
    network.load(lstm_conf['load_file_name'])
else:
    network.train(train_x, train_y)
    network.save(lstm_conf['save_file_name'])

# step 5: Predict
train_pred = network.predict(train_x)
test_pred = network.predict(test_x)

# step 6: Evaluate
evaluator = Evaluator()

# method1
train_acc = evaluator.evaluate_trend(y_true=train_y, y_pred=train_pred)
print('train=', train_acc)
test_acc = evaluator.evaluate_trend(y_true=test_y, y_pred=test_pred)
print('test=', test_acc)

# method 2
acc_train_list = evaluator.evaluate_divided_trend(train_y, train_pred)
acc_test_list = evaluator.evaluate_divided_trend(test_y, test_pred)
print('train=', acc_train_list)
print('test=', acc_test_list)

# method 3
train_acc = evaluator.evaluate_trend_simple(train_y, train_pred)
print('train=', train_acc)
test_acc = evaluator.evaluate_trend_simple(test_y, test_pred)
print('test=', test_acc)

# feature_list = lstm_conf['feature_name']
# save_feature_selection(feature_list, acc)
