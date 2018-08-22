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

cnn_conf = CNN_Config()
cnn_conf.update(use_previous_model=False)

# step 1: Get dataset (csv)
data = pd.read_csv(cnn_conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
feature_and_label_name = list(np.copy(cnn_conf['feature_name']))
feature_and_label_name.extend(cnn_conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
train_size = int(len(data) * cnn_conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_cnn(train, cnn_conf['time_step'])
test_x, test_y = data_transform_cnn(test, cnn_conf['time_step'])

indices = find_all_indices(train_y, 1)
indices.extend(find_all_indices(train_y, -1))
train_x = np.array(train_x)[indices]
train_y = np.array(train_y)[indices]

train_y = one_hot_encode(train_y, 3)
test_y = one_hot_encode(test_y, 3)

# step 4: Create and train model_weight
network = CNN(cnn_conf)
if cnn_conf['use_previous_model']:
    network.load(cnn_conf['file_name'])
else:
    network.train(train_x, train_y)
    network.save(cnn_conf['file_name'])

# step 5: Predict
train_pred = network.predict(train_x)
test_pred = network.predict(test_x)

# step 6: Evaluate
evaluator = Evaluator()

train_y = one_hot_decode(batch_labelize_prob_vector(train_y))
train_pred = one_hot_decode(batch_labelize_prob_vector(train_pred))
test_y = one_hot_decode(batch_labelize_prob_vector(test_y))
test_pred = one_hot_decode(batch_labelize_prob_vector(test_pred))
# plot_scatter(test_y, test_pred)

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
