from model.network import LSTM_MV
import keras.backend as K
from model.config import *
import warnings
import pandas as pd
from model.utils import *
from model.evaluator import Evaluator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

lstm_conf = LSTM_Config()
lstm_conf.update(use_previous_model=False,
                 load_file_name='lstm_mv.h5',
                 save_file_name='lstm_mv.h5')
lstm_conf.update(label_name=['1min_mean_price', '1min_price_std'])

# step 1: Get dataset (csv)
data = pd.read_csv(lstm_conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
feature_and_label_name = list(np.copy(lstm_conf['feature_name']))
feature_and_label_name.extend(lstm_conf['label_name'])
data = data[feature_and_label_name].values

# step 3: Preprocess
data = feature_normalize(data)
train_size = int(len(data) * lstm_conf['training_set_proportion'])
train, test = data[0:train_size, :], data[train_size:len(data), :]
train_x, train_y = data_transform_lstm_mv(train, lstm_conf['time_step'])
test_x, test_y = data_transform_lstm_mv(test, lstm_conf['time_step'])

# step 4: Create and train model_weight
network = LSTM_MV(lstm_conf)
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
print('simple evaluation')

# method1
acc = evaluator.evaluate_trend(y_true=train_y, y_pred=train_pred)
print(acc)
acc = evaluator.evaluate_trend(y_true=test_y, y_pred=test_pred)
print(acc)

# method 2
acc_train_list = evaluator.evaluate_divided_trend(train_y, train_pred)
acc_test_list = evaluator.evaluate_divided_trend(test_y, test_pred)
print('acc_train_list = ' + str(acc_train_list))
print('acc_test_list = ' + str(acc_test_list))

# method 3
acc = evaluator.evaluate_trend_simple(train_y, train_pred)
print(acc)
acc = evaluator.evaluate_trend_simple(test_y, test_pred)
print(acc)
