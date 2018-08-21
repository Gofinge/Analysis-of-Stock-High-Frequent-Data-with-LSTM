from HF.config import *
from HF.network import *
from HF.utils import *
from HF.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

cnn_conf = Config()
cnn_conf.update(feature_name=['buy5', 'bc5', 'buy4', 'bc4', 'buy3', 'bc3', 'buy2', 'bc2', 'buy1', 'bc1',
                               'sale1', 'sc1', 'sale2', 'sc2', 'sale3', 'sc3', 'sale4', 'sc4', 'sale5', 'sc5'])
cnn_conf.update(feature_num=20)

# step 1: Get dataset (csv)
file_path = os.getcwd()[:-5] + cnn_conf['data_file_path']
data = pd.read_csv(file_path, encoding='gbk')

# step 2: Select Feature
feature_and_label_name = cnn_conf['feature_name']
feature_and_label_name.extend(cnn_conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
data = feature_normalize(data)
train_size = int(len(data) * cnn_conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_cnn(train, cnn_conf['time_step'])
test_x, test_y = data_transform_cnn(test, cnn_conf['time_step'])

# step 4: Create and train model
network = CNN(cnn_conf)
model_file_name = os.getcwd()[:-5] + cnn_conf['file_name']
if cnn_conf['use_previous_model']:
    network.load(model_file_name)
else:
    network.train(train_x, train_y)
    network.save(model_file_name)

# step 5: Predict
train_pred = network.predict(train_x)
test_pred = network.predict(test_x)

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

# method 3
print('simple evaluation')
acc = evaluator.evaluate_trend_simple(y_true=train_y, y_pred=train_pred)
print(acc)
acc = evaluator.evaluate_trend_simple(y_true=test_y, y_pred=test_pred)
print(acc)

# step 7: Plot
plt.figure(figsize=(200, 15))
plt.plot(test_y)
plt.plot(test_pred)
plt.show()
