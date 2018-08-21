from sklearn import linear_model
from HF.config import *
import pandas as pd
from HF.utils import *
import matplotlib.pyplot as plt


conf = LM_Config()

reg = linear_model.Lasso(alpha=0.1)

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

reg.fit(train_x, train_y)
pred = reg.predict(test_x)
print(pred)

plt.plot(pred[0:30])
plt.plot(test_y[0:30])
plt.show()