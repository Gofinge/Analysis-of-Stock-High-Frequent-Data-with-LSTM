from model.config import *
import matplotlib.pyplot as plt
import pandas as pd


conf = LSTM_Config()

dataSet = pd.read_csv(conf['data_file_path'], encoding='gbk')

mid_price = dataSet['mid_price'].values - 8.1
mid_price_delta = dataSet['mid_price_delta'].values

plt.figure(figsize=(200, 15))
plt.plot(mid_price)
plt.plot(mid_price_delta)
plt.legend(['mid_price - 8.1', 'mid_price_delta'], loc='upper right')
plt.title('mid_price & mid_price_delta')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
