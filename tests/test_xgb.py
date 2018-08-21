from HF.config import *
from HF.network import *
from HF.utils import *
from HF.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd
import xgboost as xgb
from HF.classifier import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

conf = lstm_config()

# step 1: Get dataset (csv)
data = pd.read_csv('SH600031_18.6.15-18.6.20.csv', encoding='gbk')

# step 2: Select Feature
feature_and_label_name = conf['feature_name']
feature_and_label_name.extend(conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
data = feature_normalize(data)
train_size = int(len(data) * conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_lstm(train, conf['time_step'])
test_x, test_y = data_transform_lstm(test, conf['time_step'])

# step 4: Create and train model
xlf = XGB()
if conf['use_previous_model']:
    xlf.load(conf['file_name'])
else:
    xlf.train(train_x, train_y, test_x, test_y)
    xlf.save(conf['file_name'])

# step 5: Predict
train_pred = xlf.predict(train_x)
print('shift + enter')
test_pred = xlf.predict(test_x)

# step 6: Evaluate
evaluator = Evaluator()

# method 1
print('evaluate trend')
total_acc, stay_acc, rise_dec_acc = evaluator.evaluate_trend(y_true=train_y, y_pred=train_pred)
print(total_acc, stay_acc, rise_dec_acc)
total_acc, stay_acc, rise_dec_acc = evaluator.evaluate_trend(y_true=test_y, y_pred=test_pred)
print(total_acc, stay_acc, rise_dec_acc)

# method 2
print('evaluate trend without stay')
acc = evaluator.evaluate_trend_without_stay(y_true=train_y, y_pred=train_pred)
print(acc)
acc = evaluator.evaluate_trend_without_stay(y_true=test_y, y_pred=test_pred)
print(acc)

# step 7: Plot
plt.figure(figsize=(200, 15))
plt.plot(test_y)
plt.plot(test_pred)
plt.show()

