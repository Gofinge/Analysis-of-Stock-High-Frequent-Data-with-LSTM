from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from model.config import *
import pandas as pd
from model.utils import *
import matplotlib.pyplot as plt
from model.evaluator import *

conf = LM_Config()

# step 1: Get dataset (csv)
data = pd.read_csv(conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
feature_and_label_name = conf['feature_name']
feature_and_label_name.extend(conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
train_size = int(len(data) * conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_for_xgboost(train)
test_x, test_y = data_transform_for_xgboost(test)

classify = False
if classify:
    train_y = sign(train_y)
    test_y = sign(test_y)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4)
else:
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=5)

clf = clf.fit(train_x, train_y)

clf.fit(train_x, train_y)
pred = clf.predict(test_x)

sample_size = 50
x_list = []
pred_list = []
true_list = []
for i in range(sample_size):
    if test_y[i] != 0:
        x_list.append(i)
        pred_list.append(pred[i])
        true_list.append(test_y[i])
plt.scatter(x_list, true_list)
plt.scatter(x_list, pred_list, marker='x')
plt.show()

evaluator = Evaluator()

train_pred = clf.predict(train_x)
test_pred = clf.predict(test_x)

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
