from model.config import *
import matplotlib.pyplot as plt
import pandas as pd


conf = LSTM_Config()

# step 1: Get dataset (csv)
data = pd.read_csv(conf['data_file_path'], encoding='gbk')

# step 2: Select Feature
price = data['price'].values
mid_price = data['mid_price'].values
Avg_price = data['VW_Avg_price'].values

plt.figure(figsize=(20, 20))
plt.plot(price[0:500])
plt.plot(mid_price[0:500], linewidth=3)
plt.plot(Avg_price[0:500], linewidth=3)
plt.legend(['deal_price', 'mid_price', 'Avg_price'], loc='upper right')
plt.title('deal price & mid_price & average_price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()