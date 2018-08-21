from HF.config import *
from HF.utils import *
from HF.evaluator import Evaluator
from model.config import *
from model.network import *
from model.utils import *
from model.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import xgboost as xgb
from HF.classifier import *
from sklearn import preprocessing
from model.classifier import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

conf = Config()
conf = LSTM_Config()

# step 1: Get dataset (csv)
data = pd.read_csv(conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
feature_and_label_name = conf['feature_name']
feature_and_label_name.extend(conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
data = feature_normalize(data)
train_size = int(len(data) * conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_for_xgboost(train)
test_x, test_y = data_transform_for_xgboost(test)
train_y = sign(train_y)
train_y = [train_y[i] + 1 for i in range(len(train_y))]
test_y = [test_y[i] + 1 for i in range(len(test_y))]

# lskjdfklsdjflsdkjfslkdfj
x = np.array([[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]])
y = np.array([0, 1])
t_x = np.array([[0, 1, 2, 3, 4], [2, 4, 6, 8, 10]])
t_y = np.array([0, 1])
dtrain = xgb.DMatrix(train_x, train_y)
dtest = xgb.DMatrix(test_x, test_y)

param = {
    'booster': 'gbtree',
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    # 'nthread':7,# cpu 线程数 默认最大
    'eta': 0.01,  # 如同学习率
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'max_depth': 3,  # 构建树的深度，越大越容易过拟合
    'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。

    # 'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
    'objective': 'multi:softmax',  # 二分类的问题
    'num_class': 3,  # 类别数，多分类与 multisoftmax 并用
    'seed': 1000,  # 随机种子a
    'eval_metric': 'auc'
}
model_old = xgb.train(param, dtrain, num_boost_round=100)

# step 4: Create and train model_weight

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
