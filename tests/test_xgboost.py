from model.config import *
from model.utils import *
from model.evaluator import Evaluator
from model.config import *
from model.network import *
from model.utils import *
from model.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import xgboost as xgb
from model.classifier import *
from sklearn import preprocessing
from model.classifier import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

conf = LM_Config()

# step 1: Get dataset (csv)
data = pd.read_csv(conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
data = extract_feature_and_label(data, feature_name_list=conf['feature_name'], label_name_list=conf['label_name'])

# step 3: Preprocess
train, test = divide_train_and_test(data, conf['training_set_proportion'])
train_x, train_y = data_transform_for_xgboost(train)
test_x, test_y = data_transform_for_xgboost(test)
train_y = sign(train_y)
test_y = sign(test_y)
indices = find_all_indices(train_y, 1)
indices.extend(find_all_indices(train_y, -1))
train_x = np.array(train_x)[indices]
train_y = np.array(train_y)[indices]

dtrain = xgb.DMatrix(train_x, train_y)

param = {
    'booster': 'gbtree',
    'silent': True,
    'eta': 0.01,
    'max_depth': 4,
    'gamma': 0.1,
    'objective': 'multi:softmax',
    'num_class': 3,
    'seed': 1000,
    'scale_pos_weight': 1
}

clf = xgb.XGBClassifier(**param, dtrain=dtrain)
clf.fit(train_x, train_y)
train_pred = clf.predict(train_x)
test_pred = clf.predict(test_x)

show_feature_importance(clf, conf['feature_name'])
plot_scatter(test_y, test_pred)

evaluator = Evaluator()

print('evaluate trend')
acc = evaluator.evaluate_trend(train_y, train_pred)
print(acc)
acc = evaluator.evaluate_trend(test_y, test_pred)
print(acc)

print('evaluate trend without stay')
acc = evaluator.evaluate_trend_2(train_y, train_pred)
print(acc)
acc = evaluator.evaluate_trend_2(test_y, test_pred)
print(acc)

print('simple evaluate')
acc = evaluator.evaluate_trend_simple(train_y, train_pred)
print(acc)
acc = evaluator.evaluate_trend_simple(test_y, test_pred)
print(acc)