from model.network import LSTM_MV
import warnings
import pandas as pd
from model.utils import *
from model.evaluator import Evaluator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()

lstm_conf = LSTM_Config()
lstm_conf.update(use_previous_model=1,
                 load_file_name='lstm_mv.h5',
                 save_file_name='lstm_mv.h5')
lstm_conf.update(label_name=['30s_mean_price', 'price', '30s_mean_price_delta', '30s_price_std'])

# step 1: Get dataset (csv)
data = pd.read_csv(lstm_conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
data = extract_feature_and_label(data, lstm_conf['feature_name'], lstm_conf['label_name'])

# step 3: Preprocess
# data = feature_normalize(data, 4)
train, test = divide_train_and_test(data, lstm_conf['training_set_proportion'])
train_x, train_y, train_price, train_mean_price = data_transform_lstm_mv(train, lstm_conf['time_step'])
test_x, test_y, test_price, test_mean_price = data_transform_lstm_mv(test, lstm_conf['time_step'])

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

for i in range(len(train_pred[0])):
    train_pred[0][i] += train_price[i]
for i in range(len(test_pred[0])):
    test_pred[0][i] += test_price[i]

# step 6: Evaluate
evaluator = Evaluator()
train_acc = evaluator.evaluate_mean_and_variance(train_mean_price, train_pred)
print('train=', train_acc)
test_acc = evaluator.evaluate_mean_and_variance(test_mean_price, test_pred)
print('test=', test_acc)

plot_confidence_interval(test_mean_price, test_price, test_pred[0], test_pred[1], 3000)
