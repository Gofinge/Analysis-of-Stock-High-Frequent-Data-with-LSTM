import xgboost as xgb
from .config import *


class XGB:
    def __init__(self):
        self._param = {
            'max_depth': 5,
            'eta': 0.3,
            'silent': 0,
            'objective': 'multi:softmax',
            'num_class': 3,
            'nthread': 4}
        self._xlf = xgb.XGBRegressor(**self._param)

    def save(self, file_name):
        self._xlf.save_model(file_name)

    def load(self, file_name):
        self._xlf.load_model(file_name)

    def train(self, train_x, train_y, test_x, test_y):
        self._xlf.fit(train_x, train_y, eval_set=[test_x, test_y])

    def predict(self, x):
        return self._xlf.predict(x)
